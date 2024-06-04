import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', groups=1):
        super(GatedConv1D, self).__init__()
        self.conv_gate = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, groups=groups)
        self.conv_transform = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, groups=groups)

    def forward(self, x):
        gate = torch.sigmoid(self.conv_gate(x))
        transform = self.conv_transform(x)
        return transform * gate


class SeqConvNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SeqConvNet, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=43, embedding_dim=hidden_dim)  # Adjust `num_embeddings` as needed
        
        self.gated_conv1 = GatedConv1D(hidden_dim, hidden_dim*2, 3, padding=1)
        self.gated_conv2 = GatedConv1D(hidden_dim * 2,hidden_dim * 3, 3,  padding=1)
        self.gated_conv3 = GatedConv1D(hidden_dim * 3, hidden_dim * 3, 3,  padding=1)
        
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(hidden_dim * 3, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)  # Transpose for Conv1D (N, C, L)
        
        x = self.gated_conv1(x)
        x = self.gated_conv2(x)
        x = self.gated_conv3(x)
        
        x = self.pooling(x).squeeze(-1)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.1)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.1)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, 0.1)
        x = torch.sigmoid(self.output(x))
        return x






### Hyena

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def fftconv(u, k, D):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
    
    if len(u.shape) > 3: k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)

def fftconv(u, k, D):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
    
    if len(u.shape) > 3: k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]
    """print(f"Shape of y: {y.shape}")
    print(f"Shape of u: {u.shape}")
    print(f"Shape of D: {D.shape}")"""

    if D is not None:
        """print(f"Shape of D: {D.unsqueeze(0).shape}")"""
        #out = y + u * D.unsqueeze(0)  # Unsqueeze to add batch dimension for broadcasting
        out = y + u * D.unsqueeze(-1)
    else:
        out = y
    return out.to(dtype=u.dtype)



@torch.jit.script 
def mul_sum(q, y):
    return (q * y).sum(dim=1)

class OptimModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None: optim["lr"] = lr
            if wd is not None: optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)
            

class Sin(nn.Module):
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim)) if train_freq else w * torch.ones(1, dim)

    def forward(self, x):
        return torch.sin(self.freq * x)
    
    
class PositionalEmbedding(OptimModule):
    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float=1e-5, **kwargs): 
        """Complex exponential positional embeddings for Hyena filters."""  
        super().__init__()
        
        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None] # 1, L, 1
        
        if emb_dim > 1:
            bands = (emb_dim - 1) // 2            
        # To compute the right embeddings we use the "proper" linspace 
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len # 1, L, 1 
        
        f = torch.linspace(1e-4, bands - 1, bands)[None, None] 
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register("z", z, lr=lr_pos_emb) 
        self.register("t", t, lr=0.0)
        
    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]
    

class ExponentialModulation(OptimModule):
    def __init__(
        self,
        d_model,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        modulation_lr=0.0,
        modulate: bool=True,
        shift: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register("deltas", deltas, lr=modulation_lr)
        
    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs()) 
            x = x * (decay + self.shift)
        return x                  


class HyenaFilter(OptimModule):
    def __init__(
            self, 
            d_model,
            emb_dim=3, # dim of input to MLP, augments with positional encoding
            order=16, # width of the implicit MLP 
            fused_fft_conv=False,
            seq_len=1024, 
            lr=1e-3, 
            lr_pos_emb=1e-5,
            dropout=0.0, 
            w=1, # frequency of periodic activations 
            wd=0, # weight decay of kernel parameters 
            bias=True,
            num_inner_mlps=2,
            normalized=False,
            **kwargs
        ):
        """
        Implicit long filter with modulation.
        
        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP
        """
        super().__init__()
        self.d_model = d_model
        self.use_bias = bias
        self.fused_fft_conv = fused_fft_conv
        #self.bias = nn.Parameter(torch.randn(self.d_model))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None
        self.dropout = nn.Dropout(dropout)
        
        act = Sin(dim=order, w=w)
        self.emb_dim = emb_dim
        assert emb_dim % 2 != 0 and emb_dim >= 3, "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.seq_len = seq_len
  
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)
        
        self.implicit_filter = nn.Sequential(
            nn.Linear(emb_dim, order),
            act,
        )
        for i in range(num_inner_mlps):
            self.implicit_filter.append(nn.Linear(order, order))
            self.implicit_filter.append(act)

        self.implicit_filter.append(nn.Linear(order, d_model, bias=False))
            
        self.modulation = ExponentialModulation(d_model, **kwargs)
        
        self.normalized = normalized
        for c in self.implicit_filter.children():
            for name, v in c.state_dict().items():        
                optim = {"weight_decay": wd, "lr": lr}
                setattr(getattr(c, name), "_optim", optim)

    def filter(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)
        return h

    def forward(self, x, L, k=None, bias=None, *args, **kwargs):
        if k is None: k = self.filter(L)
        
        # Ensure compatibility with filters that return a tuple 
        k = k[0] if type(k) is tuple else k 

        y = fftconv(x, k, bias)
        return y
    
    
class HyenaOperator(nn.Module):
    def __init__(
            self,
            d_model,
            l_max,
            order=2, 
            filter_order=64,
            dropout=0.0,  
            filter_dropout=0.0, 
            **filter_args,
        ):

        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        inner_width = d_model * (order + 1)
        self.dropout = nn.Dropout(dropout)
        self.in_proj = nn.Linear(d_model, inner_width)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.short_filter = nn.Conv1d(
            inner_width, 
            inner_width, 
            3,
            padding=2,
            groups=inner_width
        )
        self.filter_fn = HyenaFilter(
            d_model * (order - 1), 
            order=filter_order, 
            seq_len=l_max,
            channels=1, 
            dropout=filter_dropout, 
            **filter_args
        ) 

    def forward(self, u, *args, **kwargs):
        l = u.size(-2)
        l_filter = min(l, self.l_max)
        print(u.shape)
        u = self.in_proj(u)
        u = rearrange(u, 'b l d -> b d l')
        
        uc = self.short_filter(u)[...,:l_filter] 
        *x, v = uc.split(self.d_model, dim=1)
        
        k = self.filter_fn.filter(l_filter)[0]
        k = rearrange(k, 'l (o d) -> o d l', o=self.order - 1)
        bias = rearrange(self.filter_fn.bias, '(o d) -> o d', o=self.order - 1)
        
        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o])

        y = rearrange(v * x[0], 'b d l -> b l d')

        y = self.out_proj(y)
        return y

"""
class HyenaOperator(nn.Module):
    def __init__(
            self,
            d_model,
            l_max,
            d_out=None,
            order=2, 
            filter_order=64,
            dropout=0.0,  
            filter_dropout=0.0, 
            **filter_args,
        ):

        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        inner_width = d_model * (order + 1)
        self.dropout = nn.Dropout(dropout)
        self.in_proj = nn.Linear(d_model, inner_width)
        self.d_out = d_out if d_out else d_model
        print(f"Output dimension: {self.d_out}")
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.short_filter = nn.Conv1d(
            inner_width, 
            inner_width, 
            3,
            padding=2,
            groups=inner_width
        )
        self.filter_fn = HyenaFilter(
            d_model * (order - 1), 
            order=filter_order, 
            seq_len=l_max,
            channels=1, 
            dropout=filter_dropout, 
            **filter_args
        ) 

    def forward(self, u):
        #print(f"Input shape: {u.shape}")
        #u = u.permute(0, 2, 1)
        u = self.in_proj(u)
        #print(f"After in_proj shape: {u.shape}")
        u = rearrange(u, 'b l d -> b d l')
        
        uc = self.short_filter(u)[..., :self.l_max] 
        *x, v = uc.split(self.d_model, dim=1)
        
        k = self.filter_fn.filter(self.l_max)[0]
        k = rearrange(k, 'l (o d) -> o d l', o=self.order - 1)
        
        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(v, self.l_max, k=k[o], bias=self.filter_fn.bias)

        y = rearrange(v * x[0], 'b d l -> b l d')
        y = self.out_proj(y)
        #print(f"Output shape: {y.shape}")
        return y
"""




import torch
import torch.nn as nn
import torch.nn.functional as F


class HyenaNet(nn.Module):
    def __init__(self, vocab_size, emb_dim, seq_len, d_model, num_classes, order=2, filter_order=32):
        super(HyenaNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.hyena_operator = HyenaOperator(d_model=emb_dim, l_max=seq_len, order=order, filter_order=filter_order)
        
        # Classifier head
        self.fc1 = nn.Linear(emb_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # Convert token indices to embeddings
        #x = x.permute(0, 2, 1)  # Adjust dimension ordering for convolution
        x = self.hyena_operator(x)
        #x = x.permute(0, 2, 1)  # Adjust back after convolution
        
        x = x.mean(dim=1)  # Global average pooling over the sequence dimension
        #print(x.shape)
        # Classifier
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.1)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.1)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, 0.1)
        x = self.output(x)
        return x





# Usage
vocab_size = 43  # Total unique tokens
emb_dim = 128  # Embedding dimensionality
seq_len = 130  # Sequence length
num_classes = 3  # Output classes
#model = HyenaNet(vocab_size, emb_dim, seq_len, emb_dim, num_classes)
#print(f"Num parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

class HyenaNet3(nn.Module):
    def __init__(self, vocab_size, emb_dim, seq_len, d_model, num_classes, filter_order=64):
        super(HyenaNet3, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.hyena_operator1 = HyenaOperator(d_model=emb_dim, l_max=seq_len, d_out=emb_dim*3, filter_order=filter_order)
        self.hyena_operator2 = HyenaOperator(d_model=emb_dim*3, l_max=seq_len, d_out=emb_dim*3, filter_order=filter_order) 
        #self.hyena_operator3 = HyenaOperator(d_model=emb_dim, l_max=seq_len, d_out=emb_dim, filter_order=filter_order)
        # Classifier head
        self.fc1 = nn.Linear(emb_dim*3, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # Convert token indices to embeddings
        #x = x.permute(0, 2, 1)  # Adjust dimension ordering for convolution
        #x = self.hyena_operator(x)
        
        x = self.hyena_operator1(x)
        x = self.hyena_operator2(x)
        #x = self.hyena_operator3(x)
        #x = x.permute(0, 2, 1)  # Adjust back after convolution
        
        x = x.mean(dim=1)  # Global average pooling over the sequence dimension
        #print(x.shape)
        # Classifier
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.1)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.1)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, 0.1)
        x = self.output(x)
        return x





class HyenaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len):
        super(HyenaBlock, self).__init__()
        self.hyena_operator = HyenaOperator(d_model=in_channels, l_max=seq_len, d_out=out_channels) #, filter_order=32)

    def forward(self, x):
        return self.hyena_operator(x)

class HyenaUNet(nn.Module):
    def __init__(self, vocab_size, emb_dim, seq_len, num_classes):
        super(HyenaUNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        # Encoder pathway
        self.down1 = HyenaBlock(emb_dim, emb_dim // 2, seq_len)
        self.down2 = HyenaBlock(emb_dim // 2, emb_dim // 4, seq_len)
        self.down3 = HyenaBlock(emb_dim // 4, emb_dim // 8, seq_len)
        # Bottleneck
        self.bottleneck = HyenaBlock(emb_dim // 8, emb_dim // 16, seq_len)
        # Decoder pathway
        self.up1 = HyenaBlock(emb_dim // 16, emb_dim // 8, seq_len)
        self.up2 = HyenaBlock(emb_dim // 8, emb_dim // 4, seq_len)
        self.up3 = HyenaBlock(emb_dim // 4, emb_dim // 2, seq_len)
        # Classifier head
        self.fc1 = nn.Linear(emb_dim // 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # Embed input tokens
        # Encoder
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        # Bottleneck
        x = self.bottleneck(x)
        # Decoder
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        # Global average pooling over the sequence dimension
        x = x.mean(dim=1)
        # Classifier
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.1)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.1)
        x = self.output(x)
        return x

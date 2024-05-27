<h1>Leash-Bio Drug Target Interaction Prediction</h1>

<h2>Setup</h2>

When working on a new machine, run:

```
git clone https://paward35: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@github.com/SeanGormann/leash_dti.git
cd leash_dti/
chmod +x ./setup.sh
./setup.sh
```

<h2>Datasets </h2>

<h3>All DTI's - tran.reduced.parquet </h3>

```
wget 'https://storage.googleapis.com/kaggle-data-sets/4784530/8490074/compressed/train.reduced.parquet.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240527%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240527T090846Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=0de170910f0e58b930f3fa4864624645ec152635b2dd7a5da23effb3669864003bf7544070f7b146907430c45b9cc6259b25fbddab742dbb73751c197bc4f32d85528d194aa0e36fd58416ba2c6277d4940b32abf778c64385967ccb48b23cf12462494282eaf3032f5132d2405d075b8af914f0da1c1ace259c8e4cbe6c22fb704fc0412f6340d3e0ee97fcc3047aa3e91acf27cf3a080591777802aa813a9869527c2197d017f4a4278dbdfdd70fe12bcb52b1f7ae1b32d25c57c0adf6b7ce8738de1fd04b3cb07faff0f83eb5a9f9249f82710c494e9368cae23b7bd60f320a79763d812e9d3524a55c136ff59fe22f159aee0263ae76c9da87fdd1c0b4ae' --no-check-certificate -O all_data.zip
cd data
unzip all_data.zip
cd ..
```

<h3>5 Million</h3>

```
wget 'https://storage.googleapis.com/kaggle-data-sets/5014460/8422528/compressed/5mNegs_allp.csv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240521%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240521T091654Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=3df8af588933bb4e00aae68b6ca141af5521f0764063ad60ef17f5de8a643b8585f34e510cd19edecb45fb1c86b8f64f212e63482560020210cc9ace25b10277cf83d53e851dbaf3a184116031b93d53e4ab0ebdc4d7e9d268ef51001fcd6a89fcb8d1abe6369fa87a5eb109b7148443119b1d70affd51ba92dc343ff9facbef4bb7c10f76a3d03504aaf3ef90c4111620a6df140198589fdda2fedadebf9199764ce462c73b0c09f5ad95c12e2e635d26e8a78329026ea89d96d8510bd4467fccc0901dedab8b642003fcc8abcfc115073d08683fce0c4bed5b1d710fee13dc12cb8d20e6670e98bea5a9d28e7d0350955917e1750ac065a748fc41dece2a10' --no-check-certificate -O graph_data_30milly.zip
cd data
unzip graph_data_5milly.zip
cd ..
```

<h3>11 Million</h3>

```
wget 'https://storage.googleapis.com/kaggle-data-sets/4936645/8310370/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240507%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240507T142559Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=1ebfdf77d4a7131d53629f37a9ef06b2785758782096673e29453ad15ecf9d29158d70840ce115aac8d28134d48dc7613fcfe65fa88a7d504a2a84a53cd188db400e68e6c5666164ce38c9bc4d1efc264c8f37485571d9b80aa090074d4e22cc37284fcddddd3506e8f1620f986ad463be0ccfe2729ee1d128f7f6b9a6aa629a10de705a1bc77c0e406ea4f34a3653ba31b211c959471be7e15734abb99e9f5ea0d7dde14197a284a94da6edb7795c9be68d523ba33d2ee704bfb505c5e2a933bb87f87c6a9e5dbacb53433f8c327b4b2144bbaec29c496b1be42893a7689829157623f0bd3a9eccfa91fb44555987c0fc992363d80964d9294eb20b47e98c9c' --no-check-certificate -O graph_dataset_11m.zip
cd data
unzip graph_dataset_11m.zip
cd ..
```

<h3>30 Million</h3>

```
wget 'https://storage.googleapis.com/kaggle-data-sets/5014460/8422528/compressed/5mNegs_allp.csv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240515%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240515T161116Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=2d71f12444712c64a7d420bd709eb5ef6750d7a7011b30a7ae04b4171444354311481d87182ad315b779a1a722622a6c0b8f85d457504ae1342e62cda46320c9bc1ac068321a6b27f0ad8901e966c6c2ebd39501aa6ba719a6ec42eae2d32395fb108945b5bd44d522761672c650aa922f33325a11128073a49eda7af7ad1cc5a2cd17681fa61f7d45e9e40732303ba358578d377096549493922ad6e36a59dceaee7cdb0720fb977ae077e255601c340341909768e31b532f0c4fbbd840dbdc8dca238d4465d8d37f02ba058d85c5f07a6d89e1fc17e3ad925def750fec9e2008fd9966e783a8dd9ad5da459dbb66193970903f6e0710ee56556c54640fe881' --no-check-certificate -O graph_data_5milly.zip
cd data
unzip graph_data_30milly.zip
cd ..
```

<h2>learning </h2>

```
python main.py
```

<h2>Monitor GPU Usage</h2>

```
watch -n0.1 nvidia-smi
```

<h2>Docker Image</h2>

```
pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
```

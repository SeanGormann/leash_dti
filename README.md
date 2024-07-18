<h1>Leash-Bio Drug Target Interaction Prediction</h1>

<img src="leash-bio.png" alt="Leash-Bio Logo" style="width: 100%;">

<h2>Overview</h2>
<p>
This repository contains my contributions to the Predict New Medicines With BELKA competition hosted by Leash Biosciences on Kaggle. The goal of the competition is to use their proprietary data obtained through high-throughput screening to develop an algorithm capable of predicting the binding affinity of small molecules (drugs) to protein targets. This challenge is closely aligned with my Master's research and has long been a significant interest of mine. Moreover, this competition provided a unique opportunity to work with a massive dataset that is rarely accessible outside major pharmaceutical companies, making it an invaluable learning and research opportunity.
</p>

<h3>Why This Is Important</h3>
<p>
The pharmaceutical industry faces the enormous task of identifying effective drug molecules from a potential chemical space of 10^60 molecules. Traditional methods of drug discovery, which involve physical synthesis and testing of molecules, are time-consuming and labor-intensive. By leveraging machine learning (ML) models, we can potentially revolutionize the process of drug discovery, making it faster, more efficient, and cost-effective. Accurate prediction of drug-target interactions is crucial for the development of new treatments for various diseases, ultimately leading to improved patient outcomes and saving lives.
</p>

<h2>Competition Dates</h2>
<ul>
  <li><strong>Start Date:</strong> April 4, 2024</li>
  <li><strong>Final Submission Deadline:</strong> July 8, 2024</li>
</ul>


<h2>Approach and Strategy</h2>
<p>
My approach to tackling this problem was to use Graph Neural Networks (GNNs), which are particularly suited for representing molecular structures in ways that traditional neural networks are not. GNNs consider molecules as graphs, where atoms are nodes and bonds are edges, effectively capturing the molecular topology. The molecules were initially provided in SMILES format, which I converted into graph representations for model training, the code for which can be found in the 'preprocessing' folder.
</p>


<h3>Graph Neural Networks in Molecular Biology</h3>
<p>
GNNs have emerged as powerful tools for modeling molecular interactions due to their ability to naturally represent molecular structures. In this project, we used GNNs with 8 node features and 4 edge features, as described in the paper by Hongjie Wu et al. (2024). The paper outlines a multi-modal drug-target affinity prediction model using graph transformers and attention mechanisms (<a href="https://doi.org/10.1016/j.neunet.2023.11.018">Wu et al., 2024</a>).
</p>
<p>
Different approaches were explored for applying GNNs in this context:
</p>
<ul>
  <li><strong>Classical Message Passing:</strong> This approach involves linear transformations of node features based on their neighbors' features, effectively propagating information through the graph.</li>
  <li><strong>Gated Graph Convolutions:</strong> These convolutions incorporate gating mechanisms to control the flow of information, enhancing the model's ability to capture complex molecular interactions.</li>
  <li><strong>Attentive-FP:</strong> This method uses attention mechanisms to weigh the importance of different nodes and edges dynamically, providing a more nuanced understanding of the molecular graph.</li>
</ul>
<p>
These techniques allowed for a comprehensive analysis of molecular interactions, leveraging the strengths of GNNs to predict drug-target binding affinities accurately.
</p>


<h3>Data Handling and Model Training</h3>
<p>The provided dataset, named BELKA, included 133 million small molecules encoded in SMILES format. Given the enormity of the data, efficient handling was crucial:</p>
<ul>
  <li><strong>Data Compression:</strong> I compressed the dataset using a dictionary mapping building block IDs to graph representations, which drastically reduced memory usage and improved loading times. The target labels were stored in a compressed <code>.npz</code> file as <code>int16</code> values, further enhancing data management efficiency.</li>
  <li><strong>Cloud Computing:</strong> The training was conducted on Vastai, a cloud computing platform, utilizing a cluster of 8 GPUs. This setup provided the necessary computational power to handle the extensive data and complex model architecture, facilitating faster iteration and experimentation.</li>
</ul>



<h2>Results</h2>
<p>
The model achieved a competitive performance in predicting the binding affinity of small molecules to protein targets. The use of GNNs, combined with efficient data handling and powerful computational resources, proved to be an effective approach in this competition. We finished in the top 10% earning a bronze medal. The experience gained from this project will undoubtedly contribute to my ongoing research and future endeavors in the field of computational drug discovery.
</p>

<h2>Conclusion</h2>
<p>
Participating in the Predict New Medicines With BELKA competition was a highly rewarding experience. It provided an opportunity to apply advanced machine learning techniques to a real-world problem with significant implications for the pharmaceutical industry. I am grateful to Leash Biosciences for hosting this competition and providing such a valuable dataset. I look forward to continuing my work in this field and contributing to the development of new and effective medicines.
</p>


<h2>Technical Setup</h2>

<h3>Directory Structure</h3>
<ul>
  <li><strong>data/</strong> - Contains the processed datasets, organized for easy access during model training and evaluation.</li>
  <li><strong>models/</strong> - This directory holds all the trained models. These are version-controlled and pushed to GitHub post-training for reproducibility and collaboration.</li>
  <li><strong>preprocessing/</strong> - Scripts and utilities used for data preprocessing, including normalization, SMILES conversion, and data augmentation.</li>
  <li><strong>selfies_modelling/</strong> - Directory for models that specifically use the SELFIES representation for molecules, aiding in specialized neural network training.</li>

  <li><strong>README.md</strong> - Provides an overview of the project, setup instructions, and other essential information.</li>
  <li><strong>main.py</strong> - The main executable script that orchestrates the training and testing workflows.</li>
  <li><strong>models.py</strong> - Contains the implementation of various deep learning models used in the project, such as Graph Neural Networks.</li>
  <li><strong>requirements.txt</strong> - Lists all the dependencies required for the project, ensuring consistent setups across different environments.</li>
  <li><strong>setup.sh</strong> - Shell script for setting up the project environment, installing dependencies, and performing initial setup tasks.</li>
  <li><strong>test.py</strong> - Contains the test suite for the project, ensuring all components function correctly before deployment.</li>
  <li><strong>utils.py</strong> - Auxiliary functions supporting various operations throughout the project, including logging and data handling utilities.</li>
</ul>



<h2>Acknowledgments</h2>
<p>
I would like to thank Leash Biosciences for organizing this competition and providing the BELKA dataset. Additionally, I appreciate the support of Vastai for providing access to the necessary computational resources to complete this project and Kaggle for hosting it.
</p>

<h2>References</h2>
<p>
For further details on the competition and the dataset, please refer to the <a href="https://www.kaggle.com/competitions/leash-BELKA/overview">Kaggle competition page</a>.
</p>



<h1>Try for yourself... </h1>

<h2>Setup</h2>

When working on a new machine, run:

```
git clone https://paward35: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@github.com/SeanGormann/leash_dti.git
cd leash_dti/
chmod +x ./setup.sh
./setup.sh
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

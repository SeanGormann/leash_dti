import os
import sys
import gc
import pickle
from time import time
from pyspark.sql import SparkSession
from rdkit import Chem
from torch_geometric.data import Data
import torch

"""
The Actual Dataset in this cometition was bloated, with over 270 million rows and over 57GB. My laptob failed to even open it. It was only through some clever manipulations
that I could get her down to a still admirable 97 million rows. 

To process the data with speed and efficiency, I used PySpark to convert the SMILES to Graphs. Pyspark is a distributed computing framework that allows for streamlined processing 
and minipulation of Bid Data. It also has the added benefit of removig the need for multiprocessing and streamlines parrallel processing on multiple CPU's and multiple Compurters 
(nodes). This script is a simplified version to run on a single node.
"""

start_time = time()

# Initialize Spark Session with increased memory allocation
spark = SparkSession.builder \
    .appName("SMILES to Graph Conversion") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.memoryOverhead", "2g") \
    .getOrCreate()

# Define the SMILES to Graph conversion function
def smiles_to_graph(smiles, y_vals):
    use_efficient_storage=True   # If true, saves data as ints instead of floats, less memory footprint but will later need to be converted to floats before training, could slow down training quite a bit 
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            int(atom.IsInRing()),
            atom.GetDegree(),
            atom.GetHybridization(),
            atom.GetImplicitValence(),
            int(atom.GetIsAromatic()),
            atom.GetTotalNumHs()
        ]
        atom_features.append(features)

    hybridization = {
        Chem.rdchem.HybridizationType.SP: 1,
        Chem.rdchem.HybridizationType.SP2: 2,
        Chem.rdchem.HybridizationType.SP3: 3,
        Chem.rdchem.HybridizationType.SP3D: 4,
        Chem.rdchem.HybridizationType.SP3D2: 5,
        Chem.rdchem.HybridizationType.S: 6,
        Chem.rdchem.HybridizationType.UNSPECIFIED: 0
    }

    for i, feat in enumerate(atom_features):
        feat[4] = hybridization.get(feat[4], 0)  # Convert hybridization to int

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    if use_efficient_storage:
        node_features = torch.tensor(atom_features, dtype=torch.float16).tolist()
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().tolist()
        y_tensor = torch.tensor(y_vals, dtype=torch.uint8).tolist()
    else:
        node_features = torch.tensor(atom_features, dtype=torch.float32).tolist()
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().tolist()
        y_tensor = torch.tensor(y_vals, dtype=torch.float32).tolist()

    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'y_tensor': y_tensor
    }


# Read in data using Spark
csv_file_path = '../data/1m_samples.csv'
df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

# Convert to an RDD
rdd = df.rdd.map(lambda row: (
    row['molecule_smiles'], 
    [row['binds_BRD4'], row['binds_HSA'], row['binds_sEH']]
))

print("Converting SMILES to Graphs...")
processed_rdd = rdd.map(lambda row: smiles_to_graph(row[0], row[1])).filter(lambda x: x is not None)

# Process the data in batches to avoid memory issues
batch_size = 100_000
num_batches = (processed_rdd.count() + batch_size - 1) // batch_size
processed_data = []
print(f"Collecting the results in {num_batches} batches...")
for i in range(num_batches):
    batch = processed_rdd.zipWithIndex().filter(lambda x: i * batch_size <= x[1] < (i + 1) * batch_size).map(lambda x: x[0]).collect()
    processed_data.extend(batch)

# Save the processed data as a pickle file
save_dir = '../data/graphs'
os.makedirs(save_dir, exist_ok=True)
with open(f'{save_dir}/processed_graphs.pkl', 'wb') as f:
    pickle.dump(processed_data, f)

spark.stop()
print(f"Saved all results to {save_dir} in minutes: {((time() - start_time) / 60):.2f}.")

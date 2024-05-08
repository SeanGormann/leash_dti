import pandas as pd
import os
import sys
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import torch
from rdkit import Chem
from torch_geometric.data import Data
import gc

"""Code for preprocessing a csv file with 'molecule_smiles' str representation and converting to a graph
Room for experimentation in the graph representation

Needs to be ran in the same directory as a csv file that has the molecule smiles to be processed and the corresponding y_labels (negatives_10m.csv in this example)

Usage:
python graph_processing_v2.py 4 0 10 (4 processes, start at batch 0, process 10 batches)
"""

def smiles_to_graph(data):
    use_efficient_storage=False   #If true, saves data as ints instead of floats, less memory footprint but will later need to be converted to floats before training, could slow doesn training quite a bit 
    smiles, y_vals = data
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
        node_features = torch.tensor(atom_features, dtype=torch.int8)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        y_tensor = torch.tensor(y_vals, dtype=torch.long)  # Use long if targets are categorical IDs
    else:
        node_features = torch.tensor(atom_features, dtype=torch.float32)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        y_tensor = torch.tensor(y_vals, dtype=torch.float32)

    return Data(x=node_features, edge_index=edge_index, y=y_tensor)


def process_batch(batch_data):
    results = []
    for smiles, y_values in batch_data:
        y_tensor = torch.tensor(y_values, dtype=torch.float)
        result = smiles_to_graph(smiles, y_tensor)
        if result is not None:
            results.append(result)
    return results


total_count = 10_000_000 #98000000
#total_count = 100000 #98000000


def main(num_processes, start_batch=0, num_batches=1):
    print("Loading data...")
    #os.makedirs('graphs_alpha/all_positives', exist_ok=True)
    #os.makedirs('graphs_alpha/all_negatives', exist_ok=True)
    save_dir = 'blinded_all_data_v2/negatives' #graph_data_blind  blinded_all_data_v2/all_blind_mols
    os.makedirs(save_dir, exist_ok=True)
    csv_file_path = 'blinded_all_data_v2/negatives_10m.csv'  # Path to the CSV file #5mNegs_allp   #balanced_hsa_dataset   blinded_all_data/blind_mols.csv
    batch_size = 10_000_000//20 #100000 #325488 679_399
    print(f"Dir: {save_dir}")

    # Assuming total_count is set or calculated beforehand
    total_batches = (total_count // batch_size) + (1 if total_count % batch_size != 0 else 0)
    end_batch = min(total_batches, start_batch + num_batches)

    with Pool(processes=num_processes) as pool:
        with pd.read_csv(csv_file_path, chunksize=batch_size) as reader:
            batch_index = 0
            for df_chunk in reader:
                if batch_index >= end_batch:
                    break
                if batch_index < start_batch:
                    batch_index += 1
                    print(f"Skipping batch {batch_index}.")
                    continue

                batch_smiles = df_chunk['molecule_smiles'].tolist()

                batch_y_values = df_chunk[['binds_BRD4', 'binds_HSA', 'binds_sEH']].values.tolist()
                #df_chunk[['BRD4', 'HSA', 'sEH']] = df_chunk[['BRD4', 'HSA', 'sEH']].fillna(-1)
                #batch_y_values = df_chunk[['BRD4', 'HSA', 'sEH']].values.tolist()

                batch_data = list(zip(batch_smiles, batch_y_values))
                results = []
                for result in tqdm(pool.imap_unordered(smiles_to_graph, batch_data), total=len(batch_data), desc=f"Processing Batch {batch_index+1}"):
                    results.append(result)
                # Store the results
                torch.save(results, f'{save_dir}/graphs_batch_{batch_index+1}.pt') #blind_batch graphs_batch_
                print(f"Saved batch {batch_index+1} to {save_dir}.")
                batch_index += 1
                gc.collect()

if __name__ == '__main__':
    num_cores = multiprocessing.cpu_count() if len(sys.argv) <= 1 else int(sys.argv[1])
    start_batch = 0 if len(sys.argv) <= 2 else int(sys.argv[2])
    num_batches = 1 if len(sys.argv) <= 3 else int(sys.argv[3])
    main(num_cores, start_batch, num_batches)

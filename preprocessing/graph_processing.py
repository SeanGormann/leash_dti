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
from time import time

"""Code for preprocessing a csv file with 'molecule_smiles' str representation and converting to a graph
Room for experimentation in the graph representation

Needs to be ran in the same directory as a csv file that has the molecule smiles to be processed and the corresponding y_labels (negatives_10m.csv in this example)

Usage:
python graph_processing_v2.py 4 0 10 (4 processes, start at batch 0, process 10 batches)
"""

def smiles_to_graph(data):
    use_efficient_storage=True   #If true, saves data as ints instead of floats, less memory footprint but will later need to be converted to floats before training, could slow doesn training quite a bit 
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
        node_features = torch.tensor(atom_features, dtype=torch.float16)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        y_tensor = torch.tensor(y_vals, dtype=torch.uint8)
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


total_count = 1_000_000


def main(num_processes, start_batch=0, num_batches=1, save_in_batches=True):
    start_time = time()
    print("Loading data...")
    save_dir = '../data/graphs' 
    os.makedirs(save_dir, exist_ok=True)
    csv_file_path = '../data/1m_samples.csv'
    batch_size = total_count // 5
    print(f"Dir: {save_dir}")

    total_batches = (total_count // batch_size) + (1 if total_count % batch_size != 0 else 0)
    end_batch = min(total_batches, start_batch + num_batches)

    all_results = []
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

                batch_data = list(zip(batch_smiles, batch_y_values))
                results = []
                for result in tqdm(pool.imap_unordered(smiles_to_graph, batch_data), total=len(batch_data), desc=f"Processing Batch {batch_index+1}"):
                    if result:
                        results.append(result)

                if save_in_batches:
                    torch.save(results, f'{save_dir}/graphs_batch_{batch_index+1}.pt')
                    print(f"Saved batch {batch_index+1} to {save_dir}.")
                else:
                    all_results.extend(results)

                batch_index += 1
                gc.collect()

    if not save_in_batches:
        torch.save(all_results, f'{save_dir}/all_graphs.pt')
        print(f"Saved all results to {save_dir} in minutes: {((time() - start_time) / 60):.2f}.")



if __name__ == '__main__':
    num_cores = multiprocessing.cpu_count() if len(sys.argv) <= 1 else int(sys.argv[1])
    start_batch = 0 if len(sys.argv) <= 2 else int(sys.argv[2])
    num_batches = 1 if len(sys.argv) <= 3 else int(sys.argv[3])
    save_in_batches = True if len(sys.argv) <= 4 else sys.argv[4].lower() == 'true'
    main(num_cores, start_batch, num_batches, save_in_batches)




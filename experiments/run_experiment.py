import argparse
import os
import h5py
import numpy as np
from rpforest.rpforest import RPForest
from scipy.io import loadmat
import json
from datetime import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm
import time


MF_DINO2 = "mf_dino2"
GLOVE = "glove"
GOOAQ = "gooaq"
UNIFORM = "uniform"
LAION = "laion"
AVAILABLE_DATASETS = [MF_DINO2, GLOVE, GOOAQ, UNIFORM, LAION]

K = 20

def get_mf_dino2():
    data = np.zeros((0, 384), dtype=np.float32)

    # Load Dino2
    for i in range(100):
        fp = loadmat(f"/Volumes/Data/mf_dino2/{i}.mat")['features']
        fp /= np.linalg.norm(fp, axis=1, keepdims=True)
        data = np.vstack((data, fp))
        
    data = np.ascontiguousarray(data, dtype=np.double)
    return data

def get_uniform():
    data = np.random.normal(0, 1, size=(1_000_000, 200)).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    data = np.ascontiguousarray(data, dtype=np.double)

    return data

def get_glove():
    data = np.load("/Volumes/Data/twitter_glove/twitter_glove_100d.npy")
    data = data.astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    
    data = np.ascontiguousarray(data, dtype=np.double)
    return data

def get_gooaq():
    with h5py.File("/Volumes/Data/gooaq/benchmark-dev-gooaq.h5", "r") as f:
        data = f["train"][:]
    data = data.astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    
    data = np.ascontiguousarray(data, dtype=np.double)
    return data

def get_laion():
    file_path = '/Volumes/Data/laion/laion-10M/laion2B-en-pca96v2-n=10M.h5'

    with h5py.File(file_path, 'r') as hdf:
        data = hdf['pca96'][:]
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        data = data[:3_000_000]
        
    print(f"Pre-uniqued size: {data.shape[0]}")
    
    data = np.unique(data, axis=0)
    print(f"Post uniqued size: {data.shape[0]}")
        
    data = np.ascontiguousarray(data, dtype=np.double)

    return data

def scramble_separate_queries(data, n_queries):
    """
    Scrambles the data.
    Returns (data, queries)
    """
    np.random.shuffle(data)
    return data[n_queries:], data[:n_queries]

def do_static_query(structure, query, k):
    return structure.query(query, k)

def do_dynamic_query(structure, query, k):
    return structure.query_mtq(query, k)

def measure_recall_static_vs_moving(num_trees, data, queries, true_nns, inertia, k=K):
    """
    Measures the difference in recall between moving the query vs not.
    """
    # Build index
    x = RPForest(leaf_size=250, no_trees=num_trees)
    x.fit(data)
    
    static_recalls = []
    dynamic_recalls = []
    
    static_ids = []
    dynamic_ids = []

    ## For static RP forest
    start_time = time.time()
    for i in range(len(queries)):
        do_static_query(x, queries[i, :], k)
    static_time = time.time() - start_time
    
    for i in range(len(queries)):
        static_ids.append(do_static_query(x, queries[i, :], k))

    start_time = time.time()
    ## For dynamic RP forest
    for i in range(len(queries)):
        do_dynamic_query(x, queries[i, :], k)
    dynamic_time = time.time() - start_time
    
    for i in range(len(queries)):
        dynamic_ids.append(do_dynamic_query(x, queries[i, :], k))
    
    
    for i in range(len(queries)):
        static_recalls.append( len (np.intersect1d(true_nns[i], static_ids[i])) )
        dynamic_recalls.append( len (np.intersect1d(true_nns[i], dynamic_ids[i])) )

    # Return mean recall and qps
    return np.mean(static_recalls) / k, np.mean(dynamic_recalls) / k, len(queries) / static_time, len(queries) / dynamic_time


def brute_force_knn(q, data, k: int):
    gt_dists = np.sqrt(2 - 2 * (q @ data.T))
    gt_ids = np.argsort(gt_dists)[:k]
    return gt_ids

def run_experiments(dataset_name, n_queries, data, blocks_start, blocks_stop, step, inertia, partial_polys=None, out_path=None):    
    data, queries = scramble_separate_queries(data, n_queries)

    true_nns = np.array([brute_force_knn(q, data, K) for q in queries])

    print(f"Dataset {dataset_name} | Data Shape: {data.shape} | Queries Shape: {queries.shape}")

    static_recalls = []
    dynamic_recalls = []

    static_times = []
    dynamic_times = []

    for num_trees in tqdm(range(blocks_start, blocks_stop, step)):
        static_recall, dynamic_recall, static_time, dynamic_time = measure_recall_static_vs_moving(num_trees, data, queries, true_nns, inertia)

        static_recalls.append(static_recall)
        dynamic_recalls.append(dynamic_recall)

        static_times.append(static_time)
        dynamic_times.append(dynamic_time)
        

        
    static_recalls = np.array(static_recalls)
    dynamic_recalls = np.array(dynamic_recalls)
    static_times = np.array(static_times)
    dynamic_times = np.array(dynamic_times)
    
    
    
    if out_path is None:
        json_path = f"results/{dataset_name}"
    else:
        json_path = out_path

    os.makedirs(json_path, exist_ok=True)

    results = {}
    results["time"] = datetime.utcnow().isoformat() + "Z"
    results["n_queries"] = n_queries
    results["inertia"] = inertia
    results["dataset"] = dataset
    
    results["static_recalls"] = static_recalls.tolist()
    results["dynamic_recalls"] = dynamic_recalls.tolist()
    results["static_qps"] = static_times.tolist()
    results["dynamic_qps"] = dynamic_times.tolist()

    ## TODO Plot and save comps vs recalls...
    save_graph(json_path, static_recalls, dynamic_recalls, static_times, dynamic_times)
    
    with open(f"{json_path}/output.json", "w") as f:
        json.dump(results, f, indent=2)

def save_graph(out_path, static_recalls, dynamic_recalls, static_times, dynamic_times):
    plt.plot(static_recalls, static_times, label="RP-Forest")
    plt.plot(dynamic_recalls, dynamic_times, label="MQ-Forest")
    
    plt.yscale('log')
    plt.ylim(bottom=10)
    
    plt.grid(True, alpha=0.3)
    
    plt.xlabel("Recall")
    plt.ylabel("Queries per Second")
    
    plt.legend()
    plt.savefig(f"{out_path}/graph.png")

def get_parser():
    parser = argparse.ArgumentParser(description=f'Cross polytope k-nn search experiments. Available datasets: {AVAILABLE_DATASETS}')

    parser.add_argument('dataset', type=str,
                        help='The name of the dataset to load')
    
    parser.add_argument('--start_blocks', type=int, help='The smallest number of blocks to test', default=1)
    parser.add_argument('--end_blocks', type=int, help='The largest number of blocks to test', default=10)
    parser.add_argument('--n_queries', type=int, help='The number of queries to test with', default=250)
    parser.add_argument('--inertia', type=int, help='The inertia', default=5)
    parser.add_argument('--partial_polys', type=int, help='The number of bases of partial x-polytope, if used', default=None)
    parser.add_argument('--out_path', type=str, help='Custom output path', default=None)
    parser.add_argument('--step', type=int, help='How much to increase block size by each iter', default=1)

    return parser

if __name__ == "__main__":
    np.random.seed(1413)  

    parser = get_parser()
    args = parser.parse_args()

    dataset = args.dataset
    n_queries = args.n_queries
    start_blocks = args.start_blocks
    end_blocks = args.end_blocks
    inertia = args.inertia
    partial_polys = args.partial_polys
    out_path = args.out_path
    step = args.step

    if dataset == MF_DINO2:
        # Run the dino2 experiments
        run_experiments(dataset, n_queries, get_mf_dino2(), start_blocks, end_blocks, step, inertia, partial_polys, out_path)
    elif dataset == GLOVE:
        run_experiments(dataset, n_queries, get_glove(), start_blocks, end_blocks, step, inertia, partial_polys, out_path)
    elif dataset == GOOAQ:
        run_experiments(dataset, n_queries, get_gooaq(), start_blocks, end_blocks, step, inertia, partial_polys, out_path)
    elif dataset == LAION:
        run_experiments(dataset, n_queries, get_laion(), start_blocks, end_blocks, step, inertia, partial_polys, out_path)
    elif dataset == UNIFORM:
        run_experiments(dataset, n_queries, get_uniform(), start_blocks, end_blocks, step, inertia, partial_polys, out_path)
    else:
        raise RuntimeError(f"Unknown dataset. Datasets are: {AVAILABLE_DATASETS}")
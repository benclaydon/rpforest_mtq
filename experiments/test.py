from rpforest import RPForest
import numpy as np
import time
from scipy.io import loadmat
import time

def get_mf_dino2():
    data = np.zeros((0, 384), dtype=np.float32)

    # Load Dino2
    for i in range(100):
        fp = loadmat(f"/Volumes/Data/mf_dino2/{i}.mat")['features']
        fp /= np.linalg.norm(fp, axis=1, keepdims=True)
        data = np.vstack((data, fp))

    return data.astype(np.float32)

def get_glove():
    data = np.load("/Volumes/Data/twitter_glove/twitter_glove_100d.npy")
    data = data.astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    
    data = np.ascontiguousarray(data, dtype=np.float32)
    return data


data = get_glove()
print(f"Loaded data")

model = RPForest(leaf_size=500, no_trees=6)
model.fit(data)

print(f"Made tree")

num_queries = 50
query_indices = np.random.choice(len(data), num_queries, replace=False)

time.sleep(10)


for i, query_idx in enumerate(query_indices):
    query_vector = data[query_idx]
    
    start = time.time()
    nns = model.query(query_vector, 20)
    rpforest_time = time.time() - start
    print(f"Query {i} (index {query_idx}) - RPForest:", nns)
    print(f"Query {i} - RPForest time: {rpforest_time:.6f}s")

    start = time.time()
    nns_mtq = model.query_mtq(query_vector, 20)
    mtq_time = time.time() - start
    print(f"Query {i} (index {query_idx}) - MTQ:", nns_mtq)
    print(f"Query {i} - MTQ time: {mtq_time:.6f}s")

    # Compute true 20 nearest neighbors using brute force
    start = time.time()
    similarities = data @ query_vector
    true_nns = np.argsort(-similarities)[:20]
    brute_force_time = time.time() - start
    print(f"Query {i} (index {query_idx}) - True NNs:", true_nns)
    print(f"Query {i} - Brute force time: {brute_force_time:.6f}s")

    # Print overlap between approximate and true nearest neighbors
    overlap_rpforest = len(set(nns) & set(true_nns))
    overlap_mtq = len(set(nns_mtq) & set(true_nns))
    print(f"Query {i} - RPForest overlap: {overlap_rpforest}/20")
    print(f"Query {i} - MTQ overlap: {overlap_mtq}/20")
    print()

from rpforest import RPForest
import numpy as np
import time

def get_glove():
    data = np.load("/Volumes/Data/twitter_glove/twitter_glove_100d.npy")
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    data = np.ascontiguousarray(data)
    data = data.astype(np.double)

    return data

data = get_glove()

model = RPForest(leaf_size=250, no_trees=12)
model.fit(data)

for i in range(10):
    start = time.time()
    nns = model.query(data[i], 20)
    rpforest_time = time.time() - start
    print(f"Query {i} - RPForest:", nns)
    print(f"Query {i} - RPForest time: {rpforest_time:.6f}s")

    start = time.time()
    nns_mtq = model.query_mtq(data[i], 20)
    mtq_time = time.time() - start
    print(f"Query {i} - MTQ:", nns_mtq)
    print(f"Query {i} - MTQ time: {mtq_time:.6f}s")

    # Compute true 20 nearest neighbors using brute force
    start = time.time()
    similarities = data @ data[i]
    true_nns = np.argsort(-similarities)[:20]
    brute_force_time = time.time() - start
    print(f"Query {i} - True NNs:", true_nns)
    print(f"Query {i} - Brute force time: {brute_force_time:.6f}s")

    # Print overlap between approximate and true nearest neighbors
    overlap_rpforest = len(set(nns) & set(true_nns))
    overlap_mtq = len(set(nns_mtq) & set(true_nns))
    print(f"Query {i} - RPForest overlap: {overlap_rpforest}/20")
    print(f"Query {i} - MTQ overlap: {overlap_mtq}/20")
    print()
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

    return data.astype(np.double)

def get_glove():
    data = np.load("/Volumes/Data/twitter_glove/twitter_glove_100d.npy")
    data = data.astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    
    data = np.ascontiguousarray(data, dtype=np.double)
    return data


data = get_glove()
print(f"Loaded data")

model = RPForest(leaf_size=500, no_trees=6)
model.fit(data)


print(f"Made tree")

num_queries = 10_000
query_indices = np.random.choice(len(data), num_queries, replace=False)

time.sleep(1)

# start = time.time()

# for i, query_idx in enumerate(query_indices):
#     query_vector = data[query_idx]    
#     nns = model.query(query_vector, 20)
# rpforest_time = time.time() - start
    

# print(f"RPForest time: {rpforest_time:.6f}s")

start = time.time()

for i, query_idx in enumerate(query_indices):
    query_vector = data[query_idx]    
    nns_mtq = model.query_mtq(query_vector, 20)
    
mtq_time = time.time() - start

print(f"MQ-Forest time: {mtq_time:.6f}s")

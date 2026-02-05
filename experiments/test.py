from rpforest import RPForest
import numpy as np

def get_glove():
    data = np.load("/Volumes/Data/twitter_glove/twitter_glove_100d.npy")
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    data = np.ascontiguousarray(data)
    data = data.astype(np.double)

    return data

data = get_glove()[:1000]

model = RPForest(leaf_size=250, no_trees=3)
model.fit(data)

nns = model.query(data[0], 10)
print(nns)

nns = model.query_mtq(data[0], 10)
print(nns)
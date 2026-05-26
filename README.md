# RP-Forest with Dynamic Query Updates

## Introduction

This is a modified version of RP-Forest, modified from the original which can be found [here](https://github.com/lyst/rpforest).
It uses query modification techniques, which can be read about [here](https://arxiv.org/abs/2605.23807).

### Usage
We expose a new ```query_mtq``` function. It should be executed on a fitted RP-Forest, similarly to the standard ```query``` algorithm.
A simple example is shown below:

```
model = RPForest(leaf_size=500, no_trees=6)
model.fit(data)
approx_nns = model.query_mtq(query_vector, 20)
```

The parameters for ```query_mtq``` are:
* number (int): The number of candidates to return
* warmup (int): The number of trees to search before query modification begins
* normalise (boolean): Whether queries should be l2-normalised before execution.

### Contact
If you require any assistance or have further queries, I can be contacted via email at bc89@st-andrews.ac.uk.

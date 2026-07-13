#!/usr/bin/env python3
"""Small, deterministic benchmark for regular and MTQ query execution.

The forest is built once, then queried repeatedly with the same MTQ
parameters.  The benchmark intentionally times query calls only; forest
construction is reported separately.

Example::

    python benchmarks/bench_mtq_heap.py --queries 1000
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np


# Make direct execution from a source checkout work without requiring an
# editable install.  An installed package still works normally.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rpforest import RPForest  # noqa: E402


def normalised_random(rng, points, dim):
    """Return deterministic, contiguous float32 unit vectors."""

    vectors = rng.standard_normal((points, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.ascontiguousarray(vectors)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=1000,
                        help="number of timed query calls (default: 1000)")
    parser.add_argument("--points", type=int, default=20000,
                        help="number of indexed vectors (default: 20000)")
    parser.add_argument("--dim", type=int, default=64,
                        help="vector dimensionality (default: 64)")
    parser.add_argument("--trees", type=int, default=8,
                        help="number of RP trees (default: 8)")
    parser.add_argument("--leaf-size", type=int, default=250,
                        help="RP tree leaf size (default: 250)")
    parser.add_argument("--k", type=int, default=20,
                        help="number of results requested per query (default: 20)")
    parser.add_argument(
        "--warmup", type=int, default=6,
        help="MTQ warmup-tree count passed to query_mtq (default: 6)",
    )
    args = parser.parse_args()

    positive = ("queries", "points", "dim", "trees", "leaf_size", "k")
    for name in positive:
        if getattr(args, name) < 1:
            parser.error("--{} must be at least 1".format(name.replace("_", "-")))
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return args


def main():
    args = parse_args()

    # Separate seeds keep the indexed vectors and queries deterministic while
    # avoiding queries that are merely exact copies of indexed rows.
    data = normalised_random(np.random.RandomState(12345), args.points, args.dim)
    queries = normalised_random(np.random.RandomState(67890), args.queries, args.dim)

    print("RP-Forest f32 benchmark")
    print(
        "config: queries={queries} points={points} dim={dim} trees={trees} "
        "leaf_size={leaf} k={k} warmup={warmup}".format(
            queries=args.queries,
            points=args.points,
            dim=args.dim,
            trees=args.trees,
            leaf=args.leaf_size,
            k=args.k,
            warmup=args.warmup,
        )
    )

    # Tree construction draws hyperplanes from NumPy's global RNG.
    np.random.seed(24680)
    build_start = time.perf_counter()
    forest = RPForest(leaf_size=args.leaf_size, no_trees=args.trees)
    forest.fit(data, normalise=False)
    build_seconds = time.perf_counter() - build_start

    print("build: {:.3f} s".format(build_seconds))
    for mode in ("regular", "mtq"):
        timings = []
        checksum = 0
        for query in queries:
            start = time.perf_counter()
            if mode == "regular":
                result = forest.query(query, number=args.k, normalise=False)
            else:
                result = forest.query_mtq(
                    query,
                    number=args.k,
                    warmup=args.warmup,
                    normalise=False,
                )
            timings.append(time.perf_counter() - start)
            # Consume the result so the benchmark also verifies success.
            if len(result):
                checksum += int(result[0])

        total_seconds = sum(timings)
        mean_ms = total_seconds * 1000.0 / len(timings)
        median_ms = float(np.median(timings)) * 1000.0
        qps = len(timings) / total_seconds if total_seconds else float("inf")
        print(
            "mode={} median={:.3f} ms mean={:.3f} ms "
            "QPS={:.2f} checksum={}".format(
                mode, median_ms, mean_ms, qps, checksum
            )
        )


if __name__ == "__main__":
    main()

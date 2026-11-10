# Router and evaluation

The router scores 13 branches: raw input, eight denoisers, and four low-light enhancers. The 85-value state contains six diagnostic values, five feedback values per branch, the 13-entry admission mask, and the remaining budget. The five feedback values record whether the branch ran, classification confidence, top-two margin, clipping ratio, and fallback status.

The router compares the best unused admitted branch with the best eligible observed branch. It calls the pending tool if its score exceeds the observed score by more than 0.01 and budget remains. Raw stays eligible. Restored outputs must pass the clipping and fallback checks before selection.

The router was trained offline with binary cross-entropy and AdamW. Each target records whether a branch predicted the correct class using the paired frozen classifier. Inference uses the current observations and admission mask.

## Checkpoints and dataset

Use the router and classifier from the same seed. The [frozen manifest](../results/frozen_manifest/frozen_manifest_13tool.json) lists the paths and hashes for all three pairs, the restoration checkpoints, and the YAML files. The default inference configuration selects seed 1.

Obtain TrashNet from its [original repository](https://github.com/garythung/trashnet). Place its class directories under `data/trashnet/` so each path in [the split manifests](../data/splits/) resolves. Keep the supplied split membership and order when comparing with the reported results.

## Evaluate one model pair

From the repository root, generate the five synthetic scenes in a separate directory. This example evaluates the noise and low-light scenes with the default model pair.

```sh
python scripts/make_test_degraded.py --manifest data/splits/test.tsv --out-base data/reproduction --seed 42
python scripts/eval_iterative_router.py --manifest data/reproduction/noise/manifest_eval.tsv --manifest data/reproduction/lowlight/manifest_eval.tsv --out-dir results/reproduction_seed1 --device cuda
```

For the six-scene test, also pass `data/splits/test.tsv` and the `manifest_eval.tsv` files for `motion_blur`, `jpeg`, and `mixed` as separate `--manifest` arguments. Use a fresh output directory for each run. The checked-in results are aggregates. A full rerun requires the dataset and the checkpoints in the frozen manifest.

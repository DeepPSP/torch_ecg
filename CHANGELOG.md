# Changelog

## Unreleased

### benchmarks: modern rewrite of all eight contest examples

All benchmarks under `benchmarks/` are rewritten as polished usage
examples on modern `torch_ecg` (one branch, one PR). Net change:
133 files, +2,293 / -25,474 lines.

- modern APIs throughout: library datasets / readers / metrics instead
  of local pre-upstream copies, `seed_everything` + `worker_init_fn`
  (a given `--seed` fully determines a run), configurable
  `--num-workers`, `python -m benchmarks.train_*.train` CLIs with
  `--smoke` debug mode; the old `torch.set_default_tensor_type`
  side effects, `sys.path` fallbacks and notebook drivers removed
- stale API drift fixed along the way (removed attributes
  `class_names` / `bin_pred`, `MultiLableClassificationOutput`,
  `gen_baseline_wander` per-segment signature, reader `rec_ext`
  without dot, `scipy` / `transformers` removed helpers)
- per-benchmark notes:
  - `train_crnn_cinc2021`: 3,087 -> ~535 lines, per-lead-set model
    configs systematically generated
  - `train_crnn_cinc2020`: thin variant of the CINC2021 benchmark
    (strict data subset of CINC2021)
  - `train_unet_ludb`: mask-decoding inference verified
  - `train_multi_cpsc2019`: all three architectures smoke-trained
  - `train_hybrid_cpsc2021`: official score / CPSC2019 R-peak metric
    now from the library; `train.py` runs the (idempotent) offline
    preparation via `CPSC2021Dataset.persistence()`
  - `train_hybrid_cpsc2020`: dead Keras R-peak preprocessing chain
    (2 MB of unloadable weights) removed; both model families
    smoke-verified on a synthesized database
  - `train_crnn_cinc2023`: the never-committed `models/` folder
    finally created (`CRNN_CINC2023`); docker / ML leftovers dropped
  - `train_mtl_cinc2022`: package-relative imports across the whole
    tree (incl. `wav2vec2_hf` / `wav2vec2_ta`); entry / docker / ML
    outcome leftovers dropped
- `test/test_pipelines/*` rewritten as thin wrappers over the
  benchmarks (5,399 -> 348 lines); `test/test_pipelines` un-ignored
  in `pyproject.toml` so the pipelines run in the regular
  `run-pytest` lanes (data-downloading tests marked `db`)
- the `test-ppl`, `test-models` and `test-databases` workflows
  retired (superseded by the `run-pytest` lanes)
- `test/requirements.txt`: `einops`, `transformers` added (needed by
  the cinc2022 benchmark import chain)

### checkpoints: fix the save/load config asymmetry (0.0.32 regression)

- `CkptMixin.save` gains a `model_config` parameter: the full outer
  config to persist for `from_checkpoint`'s `cls(config=...)`
  reconstruction.  `BaseTrainer.save_checkpoint` now passes the
  trainer's `model_config`, so trainer-written checkpoints again
  contain the full config (top-level `classes`, `n_leads`, ...) as
  the 0.0.31 `torch.save` path did; the 0.0.32 safetensors path
  silently stored only the model's inner `self.config`, which broke
  loading models whose `__init__` reads e.g. `config.classes`
  (observed in the cinc2023 / cinc2024 challenge repos)
- `make_safe_globals` now emits a `RuntimeWarning` listing every
  dropped config field (non-serializable values, path-like values,
  and -- with `remove_paths=True` -- strings that happen to exist on
  disk); previously these fields vanished silently (e.g. `db_dir`)

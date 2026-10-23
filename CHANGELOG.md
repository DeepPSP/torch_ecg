# Changelog

All notable changes are documented in this file (Keep a Changelog
format). Releases up to and including 0.0.33 are preserved on the
[version history](https://torch-ecg.readthedocs.io/) page of the docs.

## Unreleased

### Changed

- Import-time side effects removed (import purity): the AHA aux-data
  module no longer leaves `future.no_silent_downcasting` flipped in the
  process-global pandas options (saved and restored around the table
  parsing); `DEFAULTS.device` is detected lazily on first access instead
  of probing the CUDA driver at import (matters for Python 3.14+
  multiprocessing workers, which re-import the package, and freezes the
  device before `CUDA_VISIBLE_DEVICES` can be set); the `TrainCfg`
  modules of the 6 datasets no longer create their
  log/checkpoint/model directories at import -- every consumer (trainer,
  loggers, `CkptMixin.save`, `http_get`) creates them on demand.
- GitHub releases now carry the **sdist** as an asset (numpy-style:
  source on GitHub, wheels stay on PyPI); previously releases attached
  no files.
- The legacy-checkpoint test asset of `test_seq_lab_net` moved from
  Google Drive to this repository's release assets (`test-assets-v1`):
  gdrive quota errors made the test flaky (`gdown` failures escape
  as-is instead of the expected `RuntimeError`).
- `import torch_ecg` no longer pulls `biosppy`/`matplotlib`: the
  biosppy-based R-peak detectors and `filter_signal` calls, and the
  `ecg_plot` matplotlib stack, import lazily at first use (measured
  2.88s -> 2.62s cold).

### Fixed

- A second race in the one-time `database_citation.csv` migration (the
  first one, the `unlink` race, was fixed in 0.0.33): when several
  processes import the package concurrently, one could `read_csv` the
  citation cache while a sibling was still writing it, crashing at
  import with a parse error (observed on the ubuntu-24.04 CI canary).
  The cache is now written atomically (temp file + `os.replace`) and a
  partially written cache is treated as empty instead of raising.
- The atomic-rename temp file of that same migration now carries the
  process id in its name: with a single shared name, a concurrent
  importer's `os.replace` consumed the temp file and the remaining
  importers' replaces failed with `FileNotFoundError` (3/6 concurrent
  importers crashed in the nightly full suite).
- `PhysioNetDataBase.version` no longer silently falls back to "1.0.0"
  when the PhysioNet website is unreachable: the latest version is
  inferred from the `s3://physionet-open` bucket layout first
  (downloading an outdated version can produce a file layout the reader
  cannot parse, e.g. LUDB 1.0.0 vs 1.0.1 -- the cause of the 2026-10-19
  nightly failures).
- The `url_` HEAD probe of the static zip URL now has a timeout and
  falls back to the `get-zip` endpoint on request errors (previously it
  could hang for minutes and crash `helper()`); the get-zip fallback is
  now actually returned instead of only being printed.
- cpsc2020 `compute_metrics` verbose mode no longer prints per-record
  entries with uninitialized counters (totals are accumulated), and a
  `v_fn`/`v_fp` print typo is fixed.
- cinc2021 `_aggregate_stats` builds its stats DataFrame pandas-3.0-safe
  (object dtype preserved for the list columns) and no longer raises
  `KeyError: 'diagnosis'` when no cached stats file exists.

### Added

- A pure-Python `boto3` S3 downloader as the fallback for machines
  without the AWS CLI (thread pool, resume via size check, prefix
  handling mirroring `aws s3 sync`); `http_get` routes `s3://` URLs to
  the awscli backend when the `aws` binary is on PATH and to boto3
  otherwise, and `PhysioNetDataBase.download` keeps using S3 in both
  cases instead of silently falling back to physionet.org.

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

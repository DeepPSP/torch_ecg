# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Functions for downloading PhysioNet data from AWS S3. It is now made the default way to download data from PhysioNet.
- Add `easydict` as a dependency for backward compatibility (loading old models using safe-mode `torch.load` with `weights_only=True`. Extra dependencies are added with `torch.serialization.add_safe_globals`).

### Changed

- Test files (in the [sample-data](sample-data) directory) are updated.
- `requires-python` is updated from `>=3.7` to `>=3.9`.
- Add keyword argument `weights_only` to `from_checkpoint` and `from_remote` methods of the models (indeed the `CkptMixin` class). The default value is `"auto"`, which means the behavior is the same as before. It checks if `torch.serialization` has `add_safe_globals` attribute. If it does, it will use safe-mode `torch.load` with `weights_only=True`. Otherwise, it will use `torch.load` with `weights_only=False`.

### Deprecated

### Removed

- Restrictions on the version of `wfdb` and `numpy` packages are removed.

### Fixed

- Fix IO issues with several PhysioNet databases.

### Security

- Models are now loaded using safe-mode `torch.load` with `weights_only=True` by default.

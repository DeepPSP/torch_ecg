# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Functions for downloading PhysioNet data from AWS S3. It is now made the default way to download data from PhysioNet.
- Add `easydict` as a dependency for backward compatibility (loading old models with safe-mode `torch.load`).

### Changed

- Test files (in the [sample-data](sample-data) directory) are updated.

### Deprecated

### Removed

- Restrictions on the version of `wfdb` and `numpy` packages are removed.

### Fixed

- Fix IO issues with several PhysioNet databases.

### Security

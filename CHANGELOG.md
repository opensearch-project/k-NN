
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.10](https://github.com/opensearch-project/k-NN/compare/main...HEAD)
### Features

### Maintenance

### Bug Fixes
* Fix double free of off-heap vectors when a native index build fails [#3621](https://github.com/opensearch-project/k-NN/pull/3621)
* Honor query cancellation and timeout in memory-optimized search [#3620](https://github.com/opensearch-project/k-NN/pull/3620)

### Refactoring

### Enhancements
* Set the CMAKE_BUILD_TYPE to Release to ensure that all targets get build with optimized code [#3622](https://github.com/opensearch-project/k-NN/pull/3622)

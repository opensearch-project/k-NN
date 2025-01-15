
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.0](https://github.com/opensearch-project/k-NN/compare/2.x...HEAD)
### Features
### Enhancements
### Bug Fixes
### Infrastructure
### Documentation
### Maintenance
### Refactoring

## [Unreleased 2.x](https://github.com/opensearch-project/k-NN/compare/2.17...2.x)
### Features
### Enhancements
- Introduced a writing layer in native engines where relies on the writing interface to process IO. (#2241)[https://github.com/opensearch-project/k-NN/pull/2241]
- Allow method parameter override for training based indices (#2290) https://github.com/opensearch-project/k-NN/pull/2290]
- Optimizes lucene query execution to prevent unnecessary rewrites (#2305)[https://github.com/opensearch-project/k-NN/pull/2305]
- Add check to directly use ANN Search when filters match all docs. (#2320)[https://github.com/opensearch-project/k-NN/pull/2320]
- Use one formula to calculate cosine similarity (#2357)[https://github.com/opensearch-project/k-NN/pull/2357]
### Bug Fixes
### Infrastructure
### Documentation
### Maintenance
### Refactoring

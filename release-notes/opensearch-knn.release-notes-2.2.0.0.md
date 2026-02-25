## Version 2.2.0.0 Release Notes

Compatible with OpenSearch 2.2.0

### Features
* Lucene Based k-NN search support([#486](https://github.com/opensearch-project/k-NN/pull/486))

### Enhancements
* Add KNN codec that is based on Lucene92 codec([#444](https://github.com/opensearch-project/k-NN/pull/444))
* Remove support for innerproduct for lucene engine([#488](https://github.com/opensearch-project/k-NN/pull/488))
* Increase max dimension to 16k for nmslib and faiss([#490](https://github.com/opensearch-project/k-NN/pull/490))

### Bug Fixes
* Reject delete model request if model is in Training([#424](https://github.com/opensearch-project/k-NN/pull/424))
* Change call to Lucene VectorSimilarityFunction.convertToScore([#487](https://github.com/opensearch-project/k-NN/pull/487))

### Infrastructure
* Add fix to flaky test in ModelDaoTests([#463](https://github.com/opensearch-project/k-NN/pull/463))
* Read BWC Version from GitHub workflow([#476](https://github.com/opensearch-project/k-NN/pull/476))
* Staging for version increment automation([#442](https://github.com/opensearch-project/k-NN/pull/442))
* Remove 1.0.0 for BWC test([#492](https://github.com/opensearch-project/k-NN/pull/492))

### Maintenance
* Bump OpenSearch version to 2.2.0([#471](https://github.com/opensearch-project/k-NN/pull/471))
* Bump Gradle version to 7.5([#472](https://github.com/opensearch-project/k-NN/pull/472))
* Bump default bwc version to 1.3.4([#477](https://github.com/opensearch-project/k-NN/pull/477))

### Refactoring
* Move engine and lib components into separate files([#438](https://github.com/opensearch-project/k-NN/pull/438))
* Refactor knn type and codecs([#439](https://github.com/opensearch-project/k-NN/pull/439))
* Move mappers to separate files([#448](https://github.com/opensearch-project/k-NN/pull/448))

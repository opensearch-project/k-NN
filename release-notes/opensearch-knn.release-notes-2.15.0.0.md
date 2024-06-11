## Version 2.15.0.0 Release Notes

Compatible with OpenSearch 2.15.0

### Features
* Use the Lucene Distance Calculation Function in Script Scoring for doing exact search [#1699](https://github.com/opensearch-project/k-NN/pull/1699)
### Enhancements
* Add KnnCircuitBreakerException and modify exception message [#1688](https://github.com/opensearch-project/k-NN/pull/1688)
* Add stats for radial search [#1684](https://github.com/opensearch-project/k-NN/pull/1684)
* Support script score when doc value is disabled and fix misusing DISI [#1696](https://github.com/opensearch-project/k-NN/pull/1696)
* Add validation for pq m parameter before training starts [#1713](https://github.com/opensearch-project/k-NN/pull/1713)
* Block delete model requests if an index uses the model [#1722](https://github.com/opensearch-project/k-NN/pull/1722)
### Bug Fixes
* Block commas in model description [#1692](https://github.com/opensearch-project/k-NN/pull/1692)
* Update threshold value after new result is added [#1715](https://github.com/opensearch-project/k-NN/pull/1715)

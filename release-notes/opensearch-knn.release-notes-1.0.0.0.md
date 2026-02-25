## Version 1.0.0.0 Release Notes

Compatible with OpenSearch 1.0.0

### Features

* Add support for L-inf distance in AKNN, custom scoring and painless scripting ([#315](https://github.com/opendistro-for-elasticsearch/k-NN/pull/315))
* Add support for inner product in ANN, custom scoring and painless ([#324](https://github.com/opendistro-for-elasticsearch/k-NN/pull/324))
* Refactor interface to support method configuration in field mapping ([#20](https://github.com/opensearch-project/k-NN/pull/20))

### Enhancements

* Change mode for jni arrays release to prevent unnecessary copy backs ([#317](https://github.com/opendistro-for-elasticsearch/k-NN/pull/317))
* Update minimum score to 0. ([#318](https://github.com/opendistro-for-elasticsearch/k-NN/pull/318))
* Expose getValue method from KNNScriptDocValues ([#339](https://github.com/opendistro-for-elasticsearch/k-NN/pull/339))
* Add extra place to increase knn graph query errors ([#26](https://github.com/opensearch-project/k-NN/pull/26))

### Bug Fixes

* Add equals and hashcode to KNNMethodContext MethodComponentContext ([#48](https://github.com/opensearch-project/k-NN/pull/48))
* Add dimension validation to ANN QueryBuilder ([#332](https://github.com/opendistro-for-elasticsearch/k-NN/pull/332))
* Change score normalization for negative raw scores ([#337](https://github.com/opendistro-for-elasticsearch/k-NN/pull/337))

### Infrastructure

* Enabled automated license header checks ([#41](https://github.com/opensearch-project/k-NN/pull/41))
* Comment out flaky test ([#54](https://github.com/opensearch-project/k-NN/pull/54))
* Update OpenSearch upstream to 1.0.0 ([#58](https://github.com/opensearch-project/k-NN/pull/58))

### Documentation

* Update template files ([#50](https://github.com/opensearch-project/k-NN/pull/50))
* Include codecov badge ([#52](https://github.com/opensearch-project/k-NN/pull/52))

### Refactoring

* Renaming RestAPIs while supporting backwards compatibility. ([#18](https://github.com/opensearch-project/k-NN/pull/18))
* Rename namespace from opendistro to opensearch ([#21](https://github.com/opensearch-project/k-NN/pull/21))
* Move constants out of index folder into common ([#320](https://github.com/opendistro-for-elasticsearch/k-NN/pull/320))
* Expose inner_product space type, refactoring SpaceTypes ([#328](https://github.com/opendistro-for-elasticsearch/k-NN/pull/328))

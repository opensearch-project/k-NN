## Version 2.7.0.0 Release Notes

Compatible with OpenSearch 2.7.0

### Enhancements

* Support .opensearch-knn-model index as system index with security enabled ([#827](https://github.com/opensearch-project/k-NN/pull/827))

### Bug Fixes

* Throw errors on model deletion failures ([#834](https://github.com/opensearch-project/k-NN/pull/834))

### Infrastructure

* Add filter type to filtering release configs ([#792](https://github.com/opensearch-project/k-NN/pull/792))
* Add CHANGELOG ([#800](https://github.com/opensearch-project/k-NN/pull/800))
* Bump byte-buddy version from 1.12.22 to 1.14.2 ([#804](https://github.com/opensearch-project/k-NN/pull/804))
* Add 2.6.0 to BWC Version Matrix (([#810](https://github.com/opensearch-project/k-NN/pull/810)))
* Bump numpy version from 1.22.x to 1.24.2 ([#811](https://github.com/opensearch-project/k-NN/pull/811))
* Update BWC Version with OpenSearch Version Bump (([#813](https://github.com/opensearch-project/k-NN/pull/813)))
* Add GitHub action for secure integ tests ([#836](https://github.com/opensearch-project/k-NN/pull/836))
* Bump byte-buddy version to 1.14.3 ([#839](https://github.com/opensearch-project/k-NN/pull/839))
* Set gradle dependency scope for common-utils to testFixturesImplementation ([#844](https://github.com/opensearch-project/k-NN/pull/844))
* Add client setting to ignore warning exceptions ([#850](https://github.com/opensearch-project/k-NN/pull/850))

### Refactoring

* Replace Map, List, and Set in org.opensearch.common.collect with java.util references ([#816](https://github.com/opensearch-project/k-NN/pull/816))

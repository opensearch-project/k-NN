## Version 3.5.0 Release Notes

Compatible with OpenSearch and OpenSearch Dashboards version 3.5.0

### Features

* Index setting to disable exact search after ANN Search with Faiss efficient filters ([#3022](https://github.com/opensearch-project/k-NN/pull/3022))
* Bulk SIMD V2 Implementation ([#3075](https://github.com/opensearch-project/k-NN/pull/3075))

### Enhancements

* Correct ef_search parameter for Lucene engine and reduce to top K ([#3037](https://github.com/opensearch-project/k-NN/pull/3037))
* Field exclusion in source indexing handling ([#3049](https://github.com/opensearch-project/k-NN/pull/3049))
* Join filter clauses of nested k-NN queries to root-parent scope ([#2990](https://github.com/opensearch-project/k-NN/pull/2990))
* Regex for derived source support ([#3031](https://github.com/opensearch-project/k-NN/pull/3031))
* Update validation for cases when k is greater than total results ([#3038](https://github.com/opensearch-project/k-NN/pull/3038))
* Include AdditionalCodecs and EnginePlugin::getAdditionalCodecs hook to allow additional Codec registration ([#3085](https://github.com/opensearch-project/k-NN/pull/3085))

### Bug Fixes

* Changed warmup seek to use long instead of int to avoid overflow ([#3067](https://github.com/opensearch-project/k-NN/pull/3067))
* Fix MOS reentrant search bug in byte index. ([#3071](https://github.com/opensearch-project/k-NN/pull/3071))
* Fix nested docs query when some child docs has no vector field present ([#3051](https://github.com/opensearch-project/k-NN/pull/3051))
* Fix patch to have a valid score conversion for BinaryCagra. ([#2983](https://github.com/opensearch-project/k-NN/pull/2983))

### Infrastructure

* Add IT and bwc test with indices containing both vector and non-vector docs ([#3064](https://github.com/opensearch-project/k-NN/pull/3064))
* Gradle ban System.loadLibrary ([#3033](https://github.com/opensearch-project/k-NN/pull/3033))
* Create build graph ([#3032](https://github.com/opensearch-project/k-NN/pull/3032))

### Maintenance

* Added new exception type to signify expected warmup behavior ([#3070](https://github.com/opensearch-project/k-NN/pull/3070))
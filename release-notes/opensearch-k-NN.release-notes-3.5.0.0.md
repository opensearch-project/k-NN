## Version 3.5.0 Release Notes

Compatible with OpenSearch and OpenSearch Dashboards version 3.5.0

### Features
* Added new exception type to signify expected warmup behavior ([#3070](https://github.com/opensearch-project/k-NN/pull/3070))
* Bulk SIMD V2 Implementation ([#3075](https://github.com/opensearch-project/k-NN/pull/3075))
* Create build graph gradle task ([#3032](https://github.com/opensearch-project/k-NN/pull/3032))
* Field exclusion in source indexing handling ([#3049](https://github.com/opensearch-project/k-NN/pull/3049))
* Index setting to disable exact search after ANN Search with Faiss efficient filters ([#3022](https://github.com/opensearch-project/k-NN/pull/3022))
* Join filter clauses of nested k-NN queries to root-parent scope ([#2990](https://github.com/opensearch-project/k-NN/pull/2990))
* Regex for derived source support ([#3031](https://github.com/opensearch-project/k-NN/pull/3031))

### Enhancements
* Add IT and bwc test with indices containing both vector and non-vector docs ([#3064](https://github.com/opensearch-project/k-NN/pull/3064))
* Correct ef_search parameter for Lucene engine and reduce to top K ([#3037](https://github.com/opensearch-project/k-NN/pull/3037))
* Gradle ban System.loadLibrary ([#3033](https://github.com/opensearch-project/k-NN/pull/3033))
* Update validation for cases when k is greater than total results ([#3038](https://github.com/opensearch-project/k-NN/pull/3038))

### Bug Fixes
* Changed warmup seek to use long instead of int to avoid overflow ([#3067](https://github.com/opensearch-project/k-NN/pull/3067))
* Fix MOS reentrant search bug in byte index. ([#3071](https://github.com/opensearch-project/k-NN/pull/3071))
* Fix nested docs query when some child docs has no vector field present ([#3051](https://github.com/opensearch-project/k-NN/pull/3051))
* Fix patch to have a valid score conversion for BinaryCagra. ([#3086](https://github.com/opensearch-project/k-NN/pull/3086))
* Include AdditionalCodecs argument to allow additional Codec registration ([#3088](https://github.com/opensearch-project/k-NN/pull/3088))
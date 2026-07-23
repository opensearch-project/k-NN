## Version 3.8.0 Release Notes

Compatible with OpenSearch and OpenSearch Dashboards version 3.8.0

### Features

* Add search request processor to automatically exclude vector fields from `_source` in KNN queries ([#3152](https://github.com/opensearch-project/k-NN/pull/3152))

### Enhancements

* Fall back to exact search when Lucene's search budget is exhausted during Memory Optimized Search ([#3354](https://github.com/opensearch-project/k-NN/pull/3354))
* Introduce BulkVectorScorer to consolidate exact-search scoring into a single reusable abstraction ([#3361](https://github.com/opensearch-project/k-NN/pull/3361))

### Bug Fixes

* Fix FAISS SQ merge failure when a segment has no live vectors due to document deletion ([#3381](https://github.com/opensearch-project/k-NN/pull/3381))
* Fix inner-product score conversion for FAISS when Memory Optimized Search is enabled ([#3369](https://github.com/opensearch-project/k-NN/pull/3369))
* Fix incorrect FP16 validation being applied to SQ encoder with `bits=1` (x32 compression) ([#3366](https://github.com/opensearch-project/k-NN/pull/3366))
* Fix score corruption in multi-segment FAISS indices with ADC due to shared query vector mutation ([#3385](https://github.com/opensearch-project/k-NN/pull/3385))
* Fix NullPointerException in nested KNN search when index contains documents without the nested object ([#3368](https://github.com/opensearch-project/k-NN/pull/3368))

### Infrastructure

* Parameterize integration tests based on compression level to ensure stability for 32x default compression ([#3416](https://github.com/opensearch-project/k-NN/pull/3416))

### Documentation

* Clarify changelog guidance to prevent incorrect release notes from stale entries ([#3380](https://github.com/opensearch-project/k-NN/pull/3380))

### Maintenance

* Upgrade to Lucene 10.5.0 ([#3411](https://github.com/opensearch-project/k-NN/pull/3411))

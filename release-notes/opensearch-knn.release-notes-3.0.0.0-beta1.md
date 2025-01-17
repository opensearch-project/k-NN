## Version 3.0.0.0-beta1 Release Notes

Compatible with OpenSearch 3.0.0.beta1

### Features
* [Remote Vector Index Build] Client polling mechanism, encoder check, method parameter retrieval [#2576](https://github.com/opensearch-project/k-NN/pull/2576)
* [Remote Vector Index Build] Move client to separate module [#2603](https://github.com/opensearch-project/k-NN/pull/2603)
* Add filter function to KNNQueryBuilder with unit tests and integration tests [#2599](https://github.com/opensearch-project/k-NN/pull/2599)
* [Lucene On Faiss] Add a new mode, memory-optimized-search enable user to run vector search on FAISS index under memory constrained environment. [#2630](https://github.com/opensearch-project/k-NN/pull/2630)
* [Remote Vector Index Build] Add metric collection for remote build process [#2615](https://github.com/opensearch-project/k-NN/pull/2615)
* [Explain API Support] Added Explain API support for Exact/ANN/Radial/Disk based KNN search on Faiss Engine [#2403] (https://github.com/opensearch-project/k-NN/pull/2403)
### Enhancements
### Bug Fixes
* Fixing bug to prevent NullPointerException while doing PUT mappings [#2556](https://github.com/opensearch-project/k-NN/issues/2556)
* Add index operation listener to update translog source [#2629](https://github.com/opensearch-project/k-NN/pull/2629)
* Add parent join support for faiss hnsw cagra [#2647](https://github.com/opensearch-project/k-NN/pull/2647)
* [Remote Vector Index Build] Fix bug to support `COSINESIMIL` space type [#2627](https://github.com/opensearch-project/k-NN/pull/2627)
* Disable doc value storage for vector field storage [#2646](https://github.com/opensearch-project/k-NN/pull/2646)
### Infrastructure
* Add github action to run ITs against remote index builder [2620](https://github.com/opensearch-project/k-NN/pull/2620)
### Maintenance
* Update minimum required CMAKE version in NMSLIB [#2635](https://github.com/opensearch-project/k-NN/pull/2635)
* Revert CMake version bump, instead add CMake policy version flag to build task to support modern CMake builds [#2645](https://github.com/opensearch-project/k-NN/pull/2645/files)
### Refactoring
* Switch derived source from field attributes to segment attribute [#2606](https://github.com/opensearch-project/k-NN/pull/2606)
* Migrate derived source from filter to mask [#2612](https://github.com/opensearch-project/k-NN/pull/2612)
* Consolidate MethodFieldMapper and LuceneFieldMapper into EngineFieldMapper [#2646](https://github.com/opensearch-project/k-NN/pull/2646)

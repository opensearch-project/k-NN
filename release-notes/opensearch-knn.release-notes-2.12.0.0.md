## Version 2.12.0.0 Release Notes

Compatible with OpenSearch 2.12.0

### Features
* Add parent join support for lucene knn [#1182](https://github.com/opensearch-project/k-NN/pull/1182)
* Add parent join support for faiss hnsw [#1398](https://github.com/opensearch-project/k-NN/pull/1398)
* Add Support for Faiss SQFP16 and enable Faiss AVX2 Optimization [#1421](https://github.com/opensearch-project/k-NN/pull/1421)
### Enhancements
* Increase Lucene max dimension limit to 16,000 [#1346](https://github.com/opensearch-project/k-NN/pull/1346)
* Tuned default values for ef_search and ef_construction for better indexing and search performance for vector search [#1353](https://github.com/opensearch-project/k-NN/pull/1353)
* Enabled Filtering on Nested Vector fields with top level filters [#1372](https://github.com/opensearch-project/k-NN/pull/1372)
* Throw proper exception to invalid k-NN query [#1380](https://github.com/opensearch-project/k-NN/pull/1380)
### Bug Fixes
* Fix use-after-free case on nmslib search path [#1305](https://github.com/opensearch-project/k-NN/pull/1305)
* Allow nested knn field mapping when train model [#1318](https://github.com/opensearch-project/k-NN/pull/1318)
* Properly designate model state for actively training models when nodes crash or leave cluster [#1317](https://github.com/opensearch-project/k-NN/pull/1317)
* Fix script score queries not getting cached [#1367](https://github.com/opensearch-project/k-NN/pull/1367)
* Fix KNNScorer to apply boost [#1403](https://github.com/opensearch-project/k-NN/pull/1403)
* Fix equals and hashCode methods for KNNQuery and KNNQueryBuilder [#1397](https://github.com/opensearch-project/k-NN/pull/1397)
* Pass correct value on IDSelectorBitmap initialization [#1444](https://github.com/opensearch-project/k-NN/pull/1444)
### Infrastructure
* Upgrade gradle to 8.4 [1289](https://github.com/opensearch-project/k-NN/pull/1289)
* Refactor security testing to install from individual components [#1307](https://github.com/opensearch-project/k-NN/pull/1307)
* Refactor integ tests that access model index [#1423](https://github.com/opensearch-project/k-NN/pull/1423)
* Fix flaky model tests [#1429](https://github.com/opensearch-project/k-NN/pull/1429)
### Maintenance
* Update developer guide to include M1 Setup [#1222](https://github.com/opensearch-project/k-NN/pull/1222)
* Upgrade urllib to 1.26.17 [#1278](https://github.com/opensearch-project/k-NN/pull/1278)
* Upgrade urllib to 1.26.18 [#1319](https://github.com/opensearch-project/k-NN/pull/1319)
* Upgrade guava to 32.1.3 [#1319](https://github.com/opensearch-project/k-NN/pull/1319)
* Bump lucene codec to 99 [#1383](https://github.com/opensearch-project/k-NN/pull/1383)
* Update spotless and eclipse dependencies [#1450](https://github.com/opensearch-project/k-NN/pull/1450)
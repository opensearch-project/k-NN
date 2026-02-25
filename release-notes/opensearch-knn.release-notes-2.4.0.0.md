## Version 2.4.0.0 Release Notes

Compatible with OpenSearch 2.4.0

### Enhancements
* Merge efficient filtering from feature branch ([#588](https://github.com/opensearch-project/k-NN/pull/588))
* add groupId to pluginzip publication ([#578](https://github.com/opensearch-project/k-NN/pull/578))
* Added sample perf-test configs for faiss-ivf, faiss-ivfpq, lucene-hnsw ([#555](https://github.com/opensearch-project/k-NN/pull/555))
* Adding OSB index specification json for lucene hnsw ([#552](https://github.com/opensearch-project/k-NN/pull/552))
* Adding k-NN engine stat ([#523](https://github.com/opensearch-project/k-NN/pull/523))

### Infrastructure
* Fixed failing unit test ([#610](https://github.com/opensearch-project/k-NN/pull/610))
* Disable Code Coverage for Windows and Mac Platforms ([#603](https://github.com/opensearch-project/k-NN/pull/603))
* Update build script to publish to maven local ([#596](https://github.com/opensearch-project/k-NN/pull/596))
* Add Windows Build.sh Related Changes in k-NN ([#595](https://github.com/opensearch-project/k-NN/pull/595))
* Add mac platform to CI ([#590](https://github.com/opensearch-project/k-NN/pull/590))
* Add windows support ([#583](https://github.com/opensearch-project/k-NN/pull/583))

### Documentation
* Replace Forum link in k-NN plugin README.md ([#540](https://github.com/opensearch-project/k-NN/pull/540))
* Update dev guide with instructions for mac ([#518](https://github.com/opensearch-project/k-NN/pull/518))

### Bug Fixes
* Fix NPE on null script context ([#560](https://github.com/opensearch-project/k-NN/pull/560))
* Add fix to fromXContent and toXContent in ModelGraveyard ([#618](https://github.com/opensearch-project/k-NN/pull/618))

### Refactoring
* Refactor kNN codec related classes ([#582](https://github.com/opensearch-project/k-NN/pull/582))
* Refactor unit tests for codec ([#562](https://github.com/opensearch-project/k-NN/pull/562))

### Maintenance
* Backport lucene changes ([#575](https://github.com/opensearch-project/k-NN/pull/575))
* Increment version to 2.4.0-SNAPSHOT ([#545](https://github.com/opensearch-project/k-NN/pull/545))
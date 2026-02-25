## Version 1.3.0.0 Release Notes

Compatible with OpenSearch 1.3.0

### Enhancements

* Add Recall Tests ([#251](https://github.com/opensearch-project/k-NN/pull/251))
* Change serialization for knn vector from single array object to collection of floats ([#253](https://github.com/opensearch-project/k-NN/pull/253))
* Add ExtensiblePlugin to KNNPlugin ([#264](https://github.com/opensearch-project/k-NN/pull/264))
* Add gradle task for running integ tests in remote cluster ([#266](https://github.com/opensearch-project/k-NN/pull/266))
* Change benchmark ingest took metric to total time ([#268](https://github.com/opensearch-project/k-NN/pull/268))
* Make doc and query count configurable in benchmark ([#270](https://github.com/opensearch-project/k-NN/pull/270))

### Bug Fixes

* Set default space type to L2 to support bwc ([#267](https://github.com/opensearch-project/k-NN/pull/267))
* [BUG FIX] Add space type default and ef search parameter in warmup ([#276](https://github.com/opensearch-project/k-NN/pull/276))
* [FLAKY TEST] Fix codec test causing CI to fail ([#277](https://github.com/opensearch-project/k-NN/pull/277))
* [BUG FIX] Fix knn index shard to get bwc engine paths ([#309](https://github.com/opensearch-project/k-NN/pull/309))

### Infrastructure

* Remove jcenter repo from build related gradle files ([#261](https://github.com/opensearch-project/k-NN/pull/261))
* Add write permissions to backport action ([#262](https://github.com/opensearch-project/k-NN/pull/262))
* Add JDK 11 to CI and docs ([#271](https://github.com/opensearch-project/k-NN/pull/271))
* [Benchmark] Remove ingest results collection ([#272](https://github.com/opensearch-project/k-NN/pull/272))
* Update backport workflow to include custom branch name ([#273](https://github.com/opensearch-project/k-NN/pull/273))
* Add CI to run every night ([#278](https://github.com/opensearch-project/k-NN/pull/278))
* Use Github App to trigger CI on backport PRs ([#288](https://github.com/opensearch-project/k-NN/pull/288))
* Add auto delete workflow for backport branches ([#289](https://github.com/opensearch-project/k-NN/pull/289))
* Updates Guava versions to address CVE ([#292](https://github.com/opensearch-project/k-NN/pull/292))
* [CODE STYLE] Switch checkstyle to spotless ([#297](https://github.com/opensearch-project/k-NN/pull/297))
* Switch main to 2.0.0-SNAPSHOT, update to Gradle 7.3.3 ([#301](https://github.com/opensearch-project/k-NN/pull/301))
* Run CI on JDK 8 ([#302](https://github.com/opensearch-project/k-NN/pull/302))
* Update numpy version to 1.22.1 ([#305](https://github.com/opensearch-project/k-NN/pull/305))

### Refactoring

* Refactor benchmark dataset format and add big ann benchmark format ([#265](https://github.com/opensearch-project/k-NN/pull/265))

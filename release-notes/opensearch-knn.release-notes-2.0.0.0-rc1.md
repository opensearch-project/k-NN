## Version 2.0.0.0-rc1 Release Notes

Compatible with OpenSearch 2.0.0-rc1

### Enhancements

* Manually add footer to engine files ([#327](https://github.com/opensearch-project/k-NN/pull/327))
* Integration with base OpenSearch 2.0 ([#328](https://github.com/opensearch-project/k-NN/pull/328))
* Remove remaining mapping type ([#335](https://github.com/opensearch-project/k-NN/pull/335))
* Adding documentation on codec versioning ([#337](https://github.com/opensearch-project/k-NN/pull/337))
* Adding clear cache test step ([#339](https://github.com/opensearch-project/k-NN/pull/339))
* Add size validation for Search Model API ([#352](https://github.com/opensearch-project/k-NN/pull/352))

### Bug Fixes

* Allow null value for params in method mappings ([#354](https://github.com/opensearch-project/k-NN/pull/354))

### Infrastructure

* Change minimum supported JDK version from 8 to 11 ([#321](https://github.com/opensearch-project/k-NN/pull/321))
* Adding jdk 17 to CI ([#322](https://github.com/opensearch-project/k-NN/pull/322))
* Adding build.version_qualifier ([#324](https://github.com/opensearch-project/k-NN/pull/324))
* Remove version from CMakeLists.txt ([#325](https://github.com/opensearch-project/k-NN/pull/325))
* Add support for knn to have qualifiers ([#329](https://github.com/opensearch-project/k-NN/pull/329))
* Applying build qualifier only to knn plugin version ([#330](https://github.com/opensearch-project/k-NN/pull/330))
* Remove hardcoding of version in knn CI ([#334](https://github.com/opensearch-project/k-NN/pull/334))
* Apply spotless on entire project ([#336](https://github.com/opensearch-project/k-NN/pull/336))
* remove hardcoded URL ([#338](https://github.com/opensearch-project/k-NN/pull/338))
* Dropping support for JDK 14 ([#344](https://github.com/opensearch-project/k-NN/pull/344))
* Rename knnlib to lib ([#345](https://github.com/opensearch-project/k-NN/pull/345))
* Update knn with dynamic version assignment ([#349](https://github.com/opensearch-project/k-NN/pull/349))
* Updated issue templates from .github ([#351](https://github.com/opensearch-project/k-NN/pull/351))
* Incremented version to 2.0-rc1 ([#363](https://github.com/opensearch-project/k-NN/pull/363))

### Refactoring

* Refactor KNNCodec to use new extension point ([#319](https://github.com/opensearch-project/k-NN/pull/319))
* Refactor BWC tests into sub project ([#359](https://github.com/opensearch-project/k-NN/pull/359))

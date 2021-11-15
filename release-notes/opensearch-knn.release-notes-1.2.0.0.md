## Version 1.2.0.0 Release Notes

Compatible with OpenSearch 1.2.0

* Add support for faiss library to plugin ([#149](https://github.com/opensearch-project/k-NN/pull/149))

### Enhancements

* Include Model Index status as part of Stats API ([#179](https://github.com/opensearch-project/k-NN/pull/179))
* Split jnis into 2 libs and add common lib ([#181](https://github.com/opensearch-project/k-NN/pull/181))
* Generalize error message set in model metadata ([#184](https://github.com/opensearch-project/k-NN/pull/184))
* Delete local references when looping over map ([#185](https://github.com/opensearch-project/k-NN/pull/185))
* Add caching of java classes/methods ([#186](https://github.com/opensearch-project/k-NN/pull/186))
* Add more helpful validation messages ([#183](https://github.com/opensearch-project/k-NN/pull/183))
* Include index model degraded status as stats for given node ([#188](https://github.com/opensearch-project/k-NN/pull/188))
* Add training stats and library initialized stats ([#191](https://github.com/opensearch-project/k-NN/pull/191))

### Bug Fixes

* Fix library compile to package openblas statically ([#153](https://github.com/opensearch-project/k-NN/pull/153))
* Add validation to check max int limit in train API ([#159](https://github.com/opensearch-project/k-NN/pull/159))
* Support source filtering for model search ([#162](https://github.com/opensearch-project/k-NN/pull/162))
* Add super call to constructors in transport ([#169](https://github.com/opensearch-project/k-NN/pull/169))
* Return 400 on failed training request ([#168](https://github.com/opensearch-project/k-NN/pull/168))
* Add validation to check max k limit ([#178](https://github.com/opensearch-project/k-NN/pull/178))
* Fix bugs in parameter passing to JNI ([#189](https://github.com/opensearch-project/k-NN/pull/189))
* Clean up strings releated to faiss feature ([#190](https://github.com/opensearch-project/k-NN/pull/190))
* Fix issue passing parameters to native libraries ([#199](https://github.com/opensearch-project/k-NN/pull/199))
* Fix parameter validation for native libraries ([#202](https://github.com/opensearch-project/k-NN/pull/202))
* Fix field validation in VectorReader ([#207](https://github.com/opensearch-project/k-NN/pull/207))

### Infrastructure

* Update workflow ([#109](https://github.com/opensearch-project/k-NN/pull/109))
* Use published daily snapshot dependencies ([#119](https://github.com/opensearch-project/k-NN/pull/119))
* Add DCO workflow check ([#120](https://github.com/opensearch-project/k-NN/pull/120))
* Update branch pattern ([#123](https://github.com/opensearch-project/k-NN/pull/123))
* Increment version on main to 1.2.0.0 ([#138](https://github.com/opensearch-project/k-NN/pull/138))
* Adding knn lib build script ([#154](https://github.com/opensearch-project/k-NN/pull/154))
* Add lib into knn zip during build ([#163](https://github.com/opensearch-project/k-NN/pull/163))
* Disable simd for arm faiss ([#166](https://github.com/opensearch-project/k-NN/pull/166))
* Add checkstyle plugin dependency ([#177](https://github.com/opensearch-project/k-NN/pull/177))
* Package openmp lib with knnlib in zip and minor fixes ([#175](https://github.com/opensearch-project/k-NN/pull/175))
* Update license and attributions for faiss addition ([#187](https://github.com/opensearch-project/k-NN/pull/187))
* Update license headers ([#194](https://github.com/opensearch-project/k-NN/pull/194))
* Update license headers in gradle files ([#201](https://github.com/opensearch-project/k-NN/pull/201))

### Documentation

* Add support for codeowners to repo ([#206](https://github.com/opensearch-project/k-NN/pull/206))

### Refactoring

* Make model id part of index ([#167](https://github.com/opensearch-project/k-NN/pull/167))
* Remove unused code from function ([#196](https://github.com/opensearch-project/k-NN/pull/196))

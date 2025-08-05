
# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.2](https://github.com/opensearch-project/k-NN/compare/main...HEAD)
### Infrastructure
* Bump JDK version to 24, gradle to 8.14 [#2792](https://github.com/opensearch-project/k-NN/pull/2792)
* Bump Faiss commit to 2929bf4 [#2815](https://github.com/opensearch-project/k-NN/pull/2815)
* Bump Faiss commit to 5617caa [#2824](https://github.com/opensearch-project/k-NN/pull/2824)
* Bump Gradle to 8.14.3 [#2828](https://github.com/opensearch-project/k-NN/pull/2828)

### Enhancements
* Add random rotation feature to binary encoder for improving recall on certain datasets [#2718](https://github.com/opensearch-project/k-NN/pull/2718)
* Asymmetric Distance Computation for binary quantized faiss indices [#2733](https://github.com/opensearch-project/k-NN/pull/2733)
* Add KNN timing info to core profiler [#2785](https://github.com/opensearch-project/k-NN/pull/2785)
* Patch for supporting nested search in IndexBinaryHNSWCagra [#2824](https://github.com/opensearch-project/k-NN/pull/2824)
* Support Asymmetric Distance Computation in Lucene-on-Faiss [#2781](https://github.com/opensearch-project/k-NN/pull/2781)

### Bug Fixes
* [Remote Vector Index Build] Don't fall back to CPU on terminal failures [#2773](https://github.com/opensearch-project/k-NN/pull/2773)
* Fix @ collision in NativeMemoryCacheKeyHelper for vector index filenames containing @ characters [#2810](https://github.com/opensearch-project/k-NN/pull/2810)

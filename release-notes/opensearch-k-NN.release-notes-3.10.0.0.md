## Version 3.10.0 Release Notes

Compatible with OpenSearch and OpenSearch Dashboards version 3.10.0

### Features
* Add support for 2-bit and 4-bit scalar quantization for `half_float` vectors ([#3603](https://github.com/opensearch-project/k-NN/pull/3603))
* Support knn queries in percolator fields using exact search ([#3556](https://github.com/opensearch-project/k-NN/pull/3556))
* Support rescoring with `expand_nested_docs` on the Lucene engine ([#3579](https://github.com/opensearch-project/k-NN/pull/3579))

### Enhancements
* Add SIMD kernels for SQ 2-bit and 4-bit distance computation during indexing for faster graph builds ([#3655](https://github.com/opensearch-project/k-NN/pull/3655))
* Avoid unnecessary FP16 → FP32 → FP16 round trip when merging `half_float` segments ([#3610](https://github.com/opensearch-project/k-NN/pull/3610))
* Honor query cancellation and timeout in memory-optimized Faiss search ([#3620](https://github.com/opensearch-project/k-NN/pull/3620))
* Remove synchronized keyword from KNNSettings instance creation to reduce lock contention on hot paths ([#3630](https://github.com/opensearch-project/k-NN/pull/3630))
* Score ADC queries from a precomputed per-query table for significantly faster 1-bit quantized scoring ([#3633](https://github.com/opensearch-project/k-NN/pull/3633))
* Send `half_float` vectors to the remote index build service as native FP16 to halve upload size ([#3608](https://github.com/opensearch-project/k-NN/pull/3608))
* Use a dedicated 1-bit SQ distance computer to restore 1-bit indexing performance ([#3652](https://github.com/opensearch-project/k-NN/pull/3652))

### Bug Fixes
* Align field visibility across k-NN search paths so excluded fields behave consistently ([#3659](https://github.com/opensearch-project/k-NN/pull/3659))
* Fix SIGSEGV in Faiss SQ HNSW build caused by 32-bit offset overflow with large segments ([#3653](https://github.com/opensearch-project/k-NN/pull/3653))
* Fix derived source returning another document's nested vectors for parents with many nested docs ([#3624](https://github.com/opensearch-project/k-NN/pull/3624))
* Fix double free of off-heap vectors when a native index build fails, preventing JVM crash ([#3621](https://github.com/opensearch-project/k-NN/pull/3621))
* Fix explain NPE for Lucene engine rescore and `expand_nested_docs` queries ([#3641](https://github.com/opensearch-project/k-NN/pull/3641))
* Fix build compatibility issues from Lucene 10.5.2 upgrade ([#3631](https://github.com/opensearch-project/k-NN/pull/3631))
* Rethrow merge abort during native index write so Lucene drops the merge cleanly ([#3651](https://github.com/opensearch-project/k-NN/pull/3651))
* Return model-not-found error when getting a model before the model index exists ([#3598](https://github.com/opensearch-project/k-NN/pull/3598))

### Infrastructure
* Use clustered test vectors and the api-snapshot builder image so remote index build ITs pass on cuVS 26.08 ([#3627](https://github.com/opensearch-project/k-NN/pull/3627))

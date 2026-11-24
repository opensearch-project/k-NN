## Version 2.19.8 Release Notes

Compatible with OpenSearch and OpenSearch Dashboards version 2.19.8

### Bug Fixes

* Align field visibility across native k-NN search, exact scoring, and explanations to use the wrapped reader's field view ([#3660](https://github.com/opensearch-project/k-NN/pull/3660))

### Infrastructure

* Fix CI builds and rolling-upgrade tests for Linux, macOS, and Java 11 compatibility ([#3663](https://github.com/opensearch-project/k-NN/pull/3663))

### Maintenance

* Upgrade JaCoCo to 0.8.16 and ASM to 9.11.1 to address vulnerable coverage dependencies ([#3670](https://github.com/opensearch-project/k-NN/pull/3670))
* Bump Guava to 33.7.2-jre and failureaccess to 1.0.3 to address reported CVEs ([#3698](https://github.com/opensearch-project/k-NN/pull/3698))

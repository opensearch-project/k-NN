# OpenSearch Benchmarks for k-NN

## Overview

This directory contains code and configurations to run k-NN benchmarking 
workloads using OpenSearch benchmarks. 

The tenets of these benchmarks:
1. Modularity
2. Reproducibility
3. Extendability

## Python Setup

TODO


## Usage

```
opensearch-benchmark execute_test --target-hosts <host>:<port> --workload-path ./workload.json --pipeline benchmark-only
```

- Target hosts - host you want to run the test against
- Workload path - path to the workload you want to run
- Pipeline - tell OSB to use external cluster (eventually we will want to provision)



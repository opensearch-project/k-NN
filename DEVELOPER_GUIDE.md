- [Developer Guide](#developer-guide)
  - [Getting Started](#getting-started)
    - [Fork OpenSearch k-NN Repo](#fork-opensearch-k-nn-repo)
    - [Install Prerequisites](#install-prerequisites)
      - [JDK 14](#jdk-14)
  - [Use an Editor](#use-an-editor)
    - [IntelliJ IDEA](#intellij-idea)
  - [Build](#build)
    - [JNI Library](#jni-library)
    - [JNI Library Artifacts](#jni-library-artifacts)
  - [Run OpenSearch k-NN](#run-opensearch-k-nn)
    - [Run Single-node Cluster Locally](#run-single-node-cluster-locally)
    - [Run Multi-node Cluster Locally](#run-multi-node-cluster-locally)
  - [Debugging](#debugging)
  - [Backwards Compatibility Testing](#backwards-compatibility-testing)
    - [Adding new tests](#adding-new-tests)
  - [Submitting Changes](#submitting-changes)

# Developer Guide

So you want to contribute code to OpenSearch k-NN? Excellent! We're glad you're here. Here's what you need to do.

## Getting Started

### Fork OpenSearch k-NN Repo

Fork [opensearch-project/OpenSearch k-NN](https://github.com/opensearch-project/k-NN) and clone locally.

Example:
```
git clone https://github.com/[your username]/OpenSearch.git
```

### Install Prerequisites

#### JDK 14

OpenSearch builds using Java 14 at a minimum. This means you must have a JDK 14 installed with the environment variable 
`JAVA_HOME` referencing the path to Java home for your JDK 14 installation, e.g. `JAVA_HOME=/usr/lib/jvm/jdk-14`.

One easy way to get Java 14 on *nix is to use [sdkman](https://sdkman.io/).

```bash
curl -s "https://get.sdkman.io" | bash
source ~/.sdkman/bin/sdkman-init.sh
sdk install java 14.0.2-open
sdk use java 14.0.2-open
```

#### CMake

The plugin requires that cmake >= 3.17.2 is installed in order to build the JNI libraries.

#### Faiss Dependencies

To build the *faiss* JNI library, you need to have openmp, lapack and blas installed. For more information on *faiss* 
dependencies, please refer to [their documentation](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).

#### Environment

Currently, the plugin only supports Linux on x64 and arm platforms.

## Use an Editor

### IntelliJ IDEA

When importing into IntelliJ you will need to define an appropriate JDK. The convention is that **this SDK should be named "14"**, and the project import will detect it automatically. For more details on defining an SDK in IntelliJ please refer to [this documentation](https://www.jetbrains.com/help/idea/sdk.html#define-sdk). Note that SDK definitions are global, so you can add the JDK from any project, or after project import. Importing with a missing JDK will still work, IntelliJ will report a problem and will refuse to build until resolved.

You can import the OpenSearch project into IntelliJ IDEA as follows.

1. Select **File > Open**
2. In the subsequent dialog navigate to the root `build.gradle` file
3. In the subsequent dialog select **Open as Project**

## Build

OpenSearch k-NN uses a [Gradle](https://docs.gradle.org/6.6.1/userguide/userguide.html) wrapper for its build. 
Run `gradlew` on Unix systems.

Build OpenSearch k-NN using `gradlew build` 

```
./gradlew build
```

### JNI Library

The plugin relies on 2 JNI libraries to perform approximate k-NN search. `./gradlew build` will first build the 
libraries before running the plugins tests. If you see errors related to library linkage failure, make sure all 
libraries are in the Java library path. 

To build the JNI Library manually, follow these steps:

```
cd jni
cmake .

# To build everything, including tests
make

# To just build the libraries
make opensearchknn_nmslib opensearchknn_nmslib
```

The libraries will be placed in the `jni/release` directory.

Our JNI uses [Google Tests](https://github.com/google/googletest) for the C++ unit testing framework. To run the tests, 
run:

```
# To run all tests
./bin/jni_test

# To run nmslib tests
./bin/jni_test --gtest_filter=Nmslib*

# To run faiss tests
./bin/jni_test --gtest_filter=Faiss*
```

### JNI Library Artifacts

We build and distribute binary library artifacts with OpenSearch. We build the library binaries in 
[this script](https://github.com/opensearch-project/k-NN/blob/main/scripts/build.sh). In it, we package the libraries 
together with an openmp shared object. For blas and lapack, we statically link them into the *faiss* library.
We use Centos 7 with g++ 4.8.5 to build. Additionally, in order to provide as much
general compatibility as possible, we compile the libraries without some of the optimized instruction sets.
For users that want to get the most out of the libraries, they should follow [this section](#jni-library)
and build the libraries from source in their production environment, so that if their environment has optimized 
instruction sets, they take advantage of them.

## Run OpenSearch k-NN

### Run Single-node Cluster Locally
Run OpenSearch k-NN using `gradlew run`.

```shell script
./gradlew run
```
That will build OpenSearch and start it, writing its log above Gradle's status message. We log a lot of stuff on startup, specifically these lines tell you that plugin is ready.
```
[2020-05-29T14:50:35,167][INFO ][o.e.h.AbstractHttpServerTransport] [runTask-0] publish_address {127.0.0.1:9200}, bound_addresses {[::1]:9200}, {127.0.0.1:9200}
[2020-05-29T14:50:35,169][INFO ][o.e.n.Node               ] [runTask-0] started
```

It's typically easier to wait until the console stops scrolling, and then run `curl` in another window to check if OpenSearch instance is running.

```bash
curl localhost:9200

{
  "name" : "runTask-0",
  "cluster_name" : "runTask",
  "cluster_uuid" : "oX_S6cxGSgOr_mNnUxO6yQ",
  "version" : {
    "number" : "1.0.0-SNAPSHOT",
    "build_type" : "tar",
    "build_hash" : "0ba0e7cc26060f964fcbf6ee45bae53b3a9941d0",
    "build_date" : "2021-04-16T19:45:44.248303Z",
    "build_snapshot" : true,
    "lucene_version" : "8.7.0",
    "minimum_wire_compatibility_version" : "6.8.0",
    "minimum_index_compatibility_version" : "6.0.0-beta1"
  }
}
```
### Run Multi-node Cluster Locally

It can be useful to test and debug on a multi-node cluster. In order to launch a 3 node cluster with the KNN plugin installed, run the following command:

```
./gradlew run -PnumNodes=3
```

In order to run the integration tests with a 3 node cluster, run this command:

```
./gradlew :integTest -PnumNodes=3
```

Integration tests can be run with remote cluster. For that run the following command and replace host/port/cluster name values with ones for the target cluster:

```
./gradlew :integTestRemote -Dtests.rest.cluster=localhost:9200 -Dtests.cluster=localhost:9200 -Dtests.clustername="integTest-0" -Dhttps=false
```

In case remote cluster is secured it's possible to pass username and password with the following command:

```
./gradlew :integTestRemote -Dtests.rest.cluster=localhost:9200 -Dtests.cluster=localhost:9200 -Dtests.clustername="integTest-0" -Dhttps=true -Duser=admin -Dpassword=admin
```

### Debugging

Sometimes it is useful to attach a debugger to either the OpenSearch cluster or the integration test runner to see what's going on. For running unit tests, hit **Debug** from the IDE's gutter to debug the tests. For the OpenSearch cluster, first, make sure that the debugger is listening on port `5005`. Then, to debug the cluster code, run:

```
./gradlew :integTest -Dcluster.debug=1 # to start a cluster with debugger and run integ tests
```

OR

```
./gradlew run --debug-jvm # to just start a cluster that can be debugged
```

The OpenSearch server JVM will connect to a debugger attached to `localhost:5005` before starting. If there are multiple nodes, the servers will connect to debuggers listening on ports `5005, 5006, ...`

To debug code running in an integration test (which exercises the server from a separate JVM), first, setup a remote debugger listening on port `8000`, and then run:

```
./gradlew :integTest -Dtest.debug=1
```

The test runner JVM will connect to a debugger attached to `localhost:8000` before running the tests.

Additionally, it is possible to attach one debugger to the cluster JVM and another debugger to the test runner. First, make sure one debugger is listening on port `5005` and the other is listening on port `8000`. Then, run:
```
./gradlew :integTest -Dtest.debug=1 -Dcluster.debug=1
```

## Backwards Compatibility Testing

The purpose of Backwards Compatibility Testing and different types of BWC tests are explained [here](https://github.com/opensearch-project/opensearch-plugins/blob/main/TESTING.md#backwards-compatibility-testing)

Use these commands to run BWC tests for k-NN:

1. Mixed cluster test: `./gradlew knnBwcCluster#mixedClusterTask -Dtests.security.manager=false`
2. Rolling upgrade tests: `./gradlew knnBwcCluster#rollingUpgradeClusterTask -Dtests.security.manager=false`
3. Full restart upgrade tests: `./gradlew knnBwcCluster#fullRestartClusterTask -Dtests.security.manager=false`
4. `./gradlew bwcTestSuite -Dtests.security.manager=false` is used to run all the above bwc tests together.

Use this command to run BWC tests for a given Backwards Compatibility Version:
```
./gradlew bwcTestSuite -Dbwc.version=1.0.0.0-SNAPSHOT
```
Here, we are testing BWC Tests with BWC version of plugin as 1.0.0.0. Make sure to add the binary file of that version in the bwc directory in resources.

### Adding new tests

Before adding any new tests to Backward Compatibility Tests, we should be aware that the tests in BWC are not independent. While creating an index, a test cannot use the same index name if it is already used in other tests. Also, adding extra operations to the existing test may impact other existing tests like graphCount. 

## Submitting Changes

See [CONTRIBUTING](CONTRIBUTING.md).

## Backports

The Github workflow in [`backport.yml`](.github/workflows/backport.yml) creates backport PRs automatically when the 
original PR with an appropriate label `backport <backport-branch-name>` is merged to main with the backport workflow 
run successfully on the PR. For example, if a PR on main needs to be backported to `1.x` branch, add a label 
`backport 1.x` to the PR and make sure the backport workflow runs on the PR along with other checks. Once this PR is 
merged to main, the workflow will create a backport PR to the `1.x` branch.

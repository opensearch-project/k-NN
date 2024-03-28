# OpenSearch K-NN Microbenchmark Suite

This directory contains the microbenchmark suite of Opensearch K-NN Plugin. It relies on [JMH](http://openjdk.java.net/projects/code-tools/jmh/).

This module draws a lot of inspiration from [Opensearch benchmarks](https://github.com/opensearch-project/OpenSearch/tree/main/benchmarks). 

## Purpose

Micro benchmarks are intended to spot performance regressions in performance-critical components.

The microbenchmark suite is also handy for ad-hoc micro benchmarks but please remove them again before merging your PR.

## Getting Started

Just run `gradlew -p micro-benchmarks run` from the project root
directory. It will build all microbenchmarks, execute them and print
the result.

## Running Microbenchmarks

Running via an IDE is not supported as the results are meaningless
because we have no control over the JVM running the benchmarks.

If you want to run a specific benchmark class like, say,
`TransferVectorsBenchmarks`, you can use `--args`:

```
gradlew -p micro-benchmarks run --args ' TransferVectorsBenchmarks'
```

Setting Heap while running the benchmarks
```
./gradlew -p micro-benchmarks run --args ' -gc true ' -Djvm.heap.size=4g
```

Everything in the `'` gets sent on the command line to JMH. The leading ` `
inside the `'`s is important. Without it parameters are sometimes sent to
gradle.

## Adding Microbenchmarks

Before adding a new microbenchmark, make yourself familiar with the JMH API. You can check our existing microbenchmarks and also the
[JMH samples](http://hg.openjdk.java.net/code-tools/jmh/file/tip/jmh-samples/src/main/java/org/openjdk/jmh/samples/).

In contrast to tests, the actual name of the benchmark class is not relevant to JMH. However, stick to the naming convention and
end the class name of a benchmark with `Benchmark`. To have JMH execute a benchmark, annotate the respective methods with `@Benchmark`.

## Tips and Best Practices

To get realistic results, you should exercise care when running benchmarks. Here are a few tips:

### Do

* Ensure that the system executing your microbenchmarks has as little load as possible. Shutdown every process that can cause unnecessary
  runtime jitter. Watch the `Error` column in the benchmark results to see the run-to-run variance.
* Ensure to run enough warmup iterations to get the benchmark into a stable state. If you are unsure, don't change the defaults.
* Avoid CPU migrations by pinning your benchmarks to specific CPU cores. On Linux you can use `taskset`.
* Fix the CPU frequency to avoid Turbo Boost from kicking in and skewing your results. On Linux you can use `cpufreq-set` and the
  `performance` CPU governor.
* Vary the problem input size with `@Param`.
* Use the integrated profilers in JMH to dig deeper if benchmark results to not match your hypotheses:
    * Add `-prof gc` to the options to check whether the garbage collector runs during a microbenchmarks and skews
      your results. If so, try to force a GC between runs (`-gc true`) but watch out for the caveats.
    * Add `-prof perf` or `-prof perfasm` (both only available on Linux) to see hotspots.
* Have your benchmarks peer-reviewed.

### Don't

* Blindly believe the numbers that your microbenchmark produces but verify them by measuring e.g. with `-prof perfasm`.
* Run more threads than your number of CPU cores (in case you run multi-threaded microbenchmark).
* Look only at the `Score` column and ignore `Error`. Instead, take countermeasures to keep `Error` low / variance explainable.

## Disassembling

Disassembling is fun! Maybe not always useful, but always fun! Generally, you'll want to install `perf` and FCML's `hsdis`.
`perf` is generally available via `apg-get install perf` or `pacman -S perf`. FCML is a little more involved. This worked
on 2020-08-01:

```
wget https://github.com/swojtasiak/fcml-lib/releases/download/v1.2.2/fcml-1.2.2.tar.gz
tar xf fcml*
cd fcml*
./configure
make
cd example/hsdis
make
sudo cp .libs/libhsdis.so.0.0.0 /usr/lib/jvm/java-14-adoptopenjdk/lib/hsdis-amd64.so
```

If you want to disassemble a single method do something like this:

```
gradlew -p micro-benchmarks run --args ' MemoryStatsBenchmark -jvmArgs "-XX:+UnlockDiagnosticVMOptions -XX:CompileCommand=print,*.yourMethodName -XX:PrintAssemblyOptions=intel"
```


If you want `perf` to find the hot methods for you then do add `-prof:perfasm`.

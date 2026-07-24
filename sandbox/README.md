# The k-NN Sandbox

The `sandbox/` tree is an incubation environment for experimental k-NN engines, algorithms, and
optimizations ("tenants"): a structured home for high-risk/high-reward ideas inside the repository,
without touching what a default build ships.

> **⚠️ Experimental**: everything under `sandbox/` and `jni/sandbox/` is experimental, is **never included
> in release artifacts**, and may change or be removed without notice.

## How the gating works

| State | What happens |
|---|---|
| Default build (`./gradlew build`, `scripts/build.sh`, releases) | The `sandbox/` tree is **not in the Gradle build graph at all** (see `settings.gradle`): not compiled, not tested, not bundled. `jni/sandbox/` is not configured by CMake (`CONFIG_SANDBOX` is `OFF` and never auto-enabled). The produced plugin is identical to one built from a tree with no `sandbox/` directory. |
| `-Pknn.sandbox.enabled=true` | Sandbox subprojects join the build; each tenant jar is bundled into the plugin zip as a **runtime-only** artifact (the root project has no compile dependency on any tenant; discovery is via `ServiceLoader`); each tenant's isolated JNI library is built. Only the tenant **jar** is bundled, so vendor anything OpenSearch does not already provide. The flag currently enables **all** tenants; per-tenant selection is planned follow-up work. |
| Release builds | There is **no release guard**: release artifacts exclude the sandbox simply because the release scripts (`scripts/build.sh`) never pass the flag, so the sandbox is not in the build graph there, exactly as in the default-build row. |

The gate is a Gradle project property (`-Pknn.sandbox.enabled=true`) rather than a JVM system property:
it is namespaced so the OpenSearch distribution build can never trip it, and `-P` is this repo's house
style for build knobs.

## The three extension points

A tenant engine plugs into the core through one SPI and behaves like a built-in engine everywhere:

1. **KNNEngine layer**: implement
   [`KNNEngineDefinition`](../src/main/java/org/opensearch/knn/index/engine/KNNEngineDefinition.java) and
   register it via `META-INF/services`. `KNNEngineRegistry` discovers it at startup and the engine becomes
   a first-class `KNNEngine`: resolvable by name in mappings, present in `KNNEngine.values()`, folded into
   the core capability sets through the generic `KNNLibrary` flags (`supportsIterativeBuild()`,
   `createsCustomSegmentFiles()`, `supportsFilters()`, ...). The core never names a tenant engine.
   A misconfigured definition (a name colliding with a built-in engine or an already-registered tenant, a
   blank name, a null library, custom segment files without a native service, or a definition that throws)
   is **skipped with a warning** at startup, deliberately, so one bad experimental jar cannot take the
   node down.
2. **JNIService layer**: implement
   [`NativeEngineService`](../src/main/java/org/opensearch/knn/index/engine/NativeEngineService.java) (the
   8-op native index lifecycle). `JNIService` routes the 8 lifecycle/search ops (init/insert/write/
   template/load/query/radiusQuery/free) to it with a single uniform check
   (`knnEngine.getNativeService() != null`); binary indexes, training, and shared index state remain
   core-only today. Adding an engine touches **zero** core dispatch code. Pure-JVM tenants skip this
   layer entirely (`nativeService()` defaults to `null`).
3. **Query layer**: declare engine-specific query parameter names in the `KNNEngineDefinition`
   (`engineSpecificQueryParameters()`) and their value rules in a
   [`KNNLibrarySearchContext`](../src/main/java/org/opensearch/knn/index/engine/KNNLibrarySearchContext.java).
   The REST and gRPC layers defer declared names to the engine-aware validation in
   `KNNQueryBuilder#doToQuery`; a name no registered engine declares is rejected at parse, exactly as
   before. On the node-to-node wire, core-known parameters ride the unchanged upstream format; declared
   engine parameters ride a version-gated appendix. If any node in the cluster is too old to carry the
   appendix, serialization **fails loudly** at the coordinator, so an engine parameter is never silently
   dropped on a multi-node hop.

These contracts are pinned in CI at two levels. The **fixture engine** in
[`sandbox/common/src/test/java/org/opensearch/knn/sandbox/fixture/`](common/src/test/java/org/opensearch/knn/sandbox/fixture/),
a complete pure-Java tenant in a handful of small classes, exercises registration, capability folding,
JNIService dispatch, and query-parameter deferral in the sandbox test run. The node-to-node wire behavior
(appendix serialization, version gating, loud failure) is pinned by the core
`MethodParametersParserTests`, which run in default-build CI on every PR.

## Anatomy of a tenant

A tenant named `acme` (lowercase, no dashes: the name becomes a package name and a JNI library suffix)
consists of the following pieces. The fixture engine is the minimal in-tree reference; the first real
tenant PR (Intel SVS) serves as the full-size worked example, including the native pieces.

### Module: `sandbox/acme/`

`settings.gradle` discovers any `sandbox/<dir>` containing a `build.gradle` when the flag is on. The
build file is ~10 lines because plugins, repositories, the dependency set (`compileOnly project(':')`
etc.), and test conventions are inherited from [`sandbox/build.gradle`](build.gradle):

```groovy
/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
description = "Experimental ACME engine for k-NN"
```

Verify: `./gradlew projects -Pknn.sandbox.enabled=true | grep acme` lists `:sandbox:acme`; without the
flag, no sandbox project appears.

### Engine definition + SPI registration

One `KNNEngineDefinition` implementation in main sources
(reference: [`FixtureEngineProvider`](common/src/test/java/org/opensearch/knn/sandbox/fixture/FixtureEngineProvider.java)):

```java
public class AcmeEngineProvider implements KNNEngineDefinition {
    public String engineName() { return "acme"; }             // "engine": "acme" in mappings
    public KNNLibrary library() { return AcmeLibrary.INSTANCE; }
    public NativeEngineService nativeService() { return nativeService; }   // null for pure-JVM tenants
    public Set<String> engineSpecificQueryParameters() { return Set.of("acme_beam_width"); }
}
```

plus the service file
`sandbox/acme/src/main/resources/META-INF/services/org.opensearch.knn.index.engine.KNNEngineDefinition`
containing the implementation's fully qualified name. Any number of tenants can register simultaneously.

### Library

A `KNNLibrary` (typically extending
[`NativeLibrary`](../src/main/java/org/opensearch/knn/index/engine/NativeLibrary.java)) declaring the
engine's methods, segment-file extension, score translation, method resolver, and capability flags
(reference: [`FixtureLibrary`](common/src/test/java/org/opensearch/knn/sandbox/fixture/FixtureLibrary.java)).
The flags are how the core folds a tenant into its behavior without knowing its name:

| Flag | What it buys |
|---|---|
| `supportsIterativeBuild()` | The memory-efficient iterative build path (`initIndex` + `insertToIndex` batches + `writeIndex`). |
| `createsCustomSegmentFiles()` | The codec writes/reads segment files with the tenant's extension and routes them back to its engine. |
| `supportsFilters()` | Pre-filtered k-NN search is allowed on the engine. |
| `supportsRadialSearch()` | Radial queries are allowed; `NativeEngineService.radiusQueryIndex` becomes reachable. |
| `supportsNestedFields()` | k-NN on nested fields is allowed on the engine. |

Value rules for declared query parameters go in the library's `KNNLibrarySearchContext`
(reference: [`FixtureSearchContext`](common/src/test/java/org/opensearch/knn/sandbox/fixture/FixtureSearchContext.java)):

```java
public Map<String, Parameter<?>> supportedMethodParameters(QueryContext ctx) {
    return Map.of("acme_beam_width", new Parameter.IntegerParameter("acme_beam_width", null, (v, c) -> v > 0));
}
```

Engine parameters do **not** go in the core `MethodParameter` enum; that enum is only for parameters the
core itself owns.

### Native service (native tenants only)

A `NativeEngineService` implementation, typically extending `AbstractNativeEngineService` from
`:sandbox:common` (which supplies descriptive `UnsupportedOperationException`s for undeclared operations),
adapts each of the 8 lifecycle ops onto a static JNI binding class. `JNIService` hands the tenant
the raw method-parameters map: the tenant handles its own tuning parameters (e.g. thread counts), and the
core does not pre-extract `INDEX_THREAD_QTY` for tenants. The binding class loads its library through
the core loader:

```java
static {
    // Picks the best available SIMD variant for the host and falls back to the plain library.
    KNNLibraryLoader.loadLibraryByVariant("opensearchknn_acme");
    initLibrary();
    KNNEngine.getEngine(AcmeConstants.ACME_ENGINE_NAME).setInitialized(true);
}
```

The variant-suffix scheme (`_avx512_spr`/`_avx512`/`_avx2`/plain) mirrors faiss's opt-levels, but shipping
variants is optional: a single unsuffixed `.so` is fully supported via the loader's fallback.

Two conventions, checked in review:

* All `System.loadLibrary` calls go through `KNNLibraryLoader`; tenants must not load native libraries
  directly.
* The `NativeEngineService` must hold no static reference that class-initializes the JNI binding class:
  registration happens at node startup, but the native library must load **lazily on first use**. The
  first native tenant PR demonstrates this pattern end to end.

### Native build (native tenants only): `jni/sandbox/acme/`

Configured only when `-DCONFIG_SANDBOX=ON` (passed automatically by `-Pknn.sandbox.enabled=true`). A
`jni/sandbox/acme/tenant.cmake`, discovered automatically by `jni/sandbox/tenants.cmake`, vendors the
tenant's own copy of its underlying library (static, PIC, pinned to an exact commit) and hands the JNI
sources to the shared helper in [`jni/sandbox/cmake/SandboxTenant.cmake`](../jni/sandbox/cmake/SandboxTenant.cmake):

```cmake
knn_sandbox_add_jni_library(opensearchknn_acme
    SOURCES        ${KNN_SANDBOX_TENANT_DIR}/src/org_opensearch_knn_sandbox_acme_AcmeService.cpp
    INCLUDE_DIRS   ${KNN_SANDBOX_TENANT_DIR}/include
    LINK_LIBRARIES acme_vendor
    DEPENDS        acme_vendor_ep)
```

The helper compiles tenant code with hidden visibility (JNI entry points stay exported via `JNIEXPORT`)
and, on Linux, links with `-Wl,--exclude-libs,ALL` so symbols from statically linked archives become
local (a guarantee that is Linux-only today; macOS/Windows are un-addressed), plus an `$ORIGIN` rpath for
any runtime `.so` shipped alongside. This matters most when the tenant embeds a different version of a
C++ library the plugin already ships (e.g. faiss): two exported `faiss::*` symbol sets in one JVM
interpose and route calls into the wrong build. One thing symbol isolation cannot contain is the
threading runtime: link OpenMP **dynamically** (`libgomp.so`) so the process shares a single runtime with
the built-in libraries, and never statically embed an OpenMP runtime (two runtimes in one JVM is a known
crash/oversubscription hazard). Verify that the dynamic table lists only `Java_*`/`JNI_*` entry points
(plus linker-synthesized symbols such as `__bss_start`/`_edata`/`_end`):

```bash
nm -D --defined-only jni/build/release/libopensearchknn_acme*.so
```

Additional native-build rules:

* **Supply chain**: anything `tenant.cmake` downloads must be checksum-pinned (`URL_HASH SHA256=...` or an
  exact git commit); a user-overridable URL requires a paired `-D..._SHA256`.
* **JNI headers**: compiling the tenant's Java generates headers under
  `sandbox/acme/build/generated-jni-headers/`; copy them into `jni/sandbox/acme/include/` (headers are
  checked in so the native build never depends on a Java compile). Note that `_` in package/class names
  mangles to `_1` in JNI symbols; the generated header gets this right.
* **faiss-based tenants**: a shared `knn_sandbox_vendor_faiss` helper packaging the static+PIC faiss
  recipe (SIMD-variant selection, BLAS/LAPACK/OpenMP re-supply) ships with the first faiss-based tenant.
* **Tenant-specific cmake flags** pass through `-Psandbox.cmake.args="-DYOUR_FLAG=value;..."`; a tenant
  never edits the root `build.gradle`.

Gradle discovers the native target by convention: a tenant with `jni/sandbox/acme/tenant.cmake` builds
`libopensearchknn_acme*.so` under `./gradlew buildJniLib -Pknn.sandbox.enabled=true`.

### Tests

Unit tests live in `sandbox/acme/src/test/...` and run via
`./gradlew :sandbox:acme:test -Pknn.sandbox.enabled=true`; the fixture test classes
([`FixtureEngineRegistrationTests`](common/src/test/java/org/opensearch/knn/sandbox/fixture/FixtureEngineRegistrationTests.java),
[`FixtureJNIServiceDispatchTests`](common/src/test/java/org/opensearch/knn/sandbox/fixture/FixtureJNIServiceDispatchTests.java),
[`FixtureQueryParamDeferralTests`](common/src/test/java/org/opensearch/knn/sandbox/fixture/FixtureQueryParamDeferralTests.java))
are ready-made templates for registration, dispatch, and deferral/engine-aware validation.

REST integration tests are auto-wired: `*IT` classes extending `KNNRestTestCase` in the tenant's test
sources are detected by the shared sandbox config, which creates a `:sandbox:acme:integTest` task whose
cluster runs the sandbox-enabled plugin zip with the JNI build dir on the node's `java.library.path`.
Nothing to copy into the tenant's build.gradle; the wiring lives once in
[`sandbox/build.gradle`](build.gradle).

To run a live node: `./gradlew run -Pknn.sandbox.enabled=true`, then map a field with
`"engine": "acme"`. On a default build the same mapping is rejected as an invalid engine (verified live:
`mapper_parsing_exception: Invalid engine: acme`), which is correct behavior: the tenant does not exist
there.

## CI

The [`sandbox-check`](../.github/workflows/sandbox-check.yml) workflow runs
`./gradlew -p sandbox check -Pknn.sandbox.enabled=true` (tests + spotless) on every PR, so tenant unit
tests join CI with zero workflow changes. It runs on all PRs (not just sandbox ones) because a core
refactor can break the SPI seam without touching `sandbox/**`.

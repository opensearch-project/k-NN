/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

// Compiled into every k-NN JNI library (see opensearch_set_common_properties in cmake/macros.cmake) to fail the
// build unless the library is built as Release. The native hot paths (distance computation, graph traversal,
// quantization) are several times slower without -O3, and a non-Release build is easy to produce silently: the
// build type used to be set only as a side effect of including NMSLIB, so FAISS-only builds got no -O flag.
//
// KNN_RELEASE_BUILD is defined by CMake only for the Release configuration. Pass -DKNN_ALLOW_NON_RELEASE_BUILD=ON
// to build another configuration (e.g. Debug) on purpose.
#if !defined(KNN_ALLOW_NON_RELEASE_BUILD)
#if !defined(KNN_RELEASE_BUILD)
#error "k-NN JNI libraries must be built with CMAKE_BUILD_TYPE=Release. Pass -DKNN_ALLOW_NON_RELEASE_BUILD=ON to build another build type on purpose."
// Catches a Release build whose optimization flags were overridden (e.g. CMAKE_CXX_FLAGS_RELEASE without -O).
#elif (defined(__GNUC__) || defined(__clang__)) && !defined(__OPTIMIZE__)
#error "k-NN JNI libraries must be compiled with optimization (e.g. -O3), but the Release flags have no -O level."
#endif
#endif

namespace {
// Keeps this translation unit non-empty.
[[maybe_unused]] constexpr int kKnnBuildGuard = 0;
}  // namespace

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gtest/gtest.h"

// jni_test is compiled with the same build type and flags as the JNI libraries, so this fails unless the native code
// is built as an optimized Release build. The libraries themselves are also guarded at compile time by
// src/knn_build_guard.cpp.
TEST(BuildConfigTest, NativeCodeIsBuiltAsOptimizedRelease) {
#if defined(KNN_ALLOW_NON_RELEASE_BUILD)
    GTEST_SKIP() << "Non-Release build explicitly allowed (-DKNN_ALLOW_NON_RELEASE_BUILD=ON)";
#else
#if !defined(KNN_RELEASE_BUILD)
    FAIL() << "Native code is not built with CMAKE_BUILD_TYPE=Release";
#endif
#if (defined(__GNUC__) || defined(__clang__)) && !defined(__OPTIMIZE__)
    FAIL() << "Native code is built as Release but without optimization (no -O level)";
#endif
#endif
}

/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

#include "jni_util.h"
#include "faiss_wrapper.h"
#include "nmslib_wrapper.h"

#include <gtest/gtest.h>

// Demonstrate some basic assertions.
TEST(FaissInitLibraryTest, BasicAssertions) {
    knn_jni::faiss_wrapper::InitLibrary();
    // Expect two strings not to be equal.
    EXPECT_STRNE("hello", "wsorld");
    // Expect equality.
    EXPECT_EQ(7 * 6, 42);
}

// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

#include "faiss_util.h"

#include <vector>

#include "gtest/gtest.h"

TEST(IDGrouperBitMapTest, BasicAssertions) {
    int ids[] = {128, 1024};
    size_t length = sizeof(ids) / sizeof(ids[0]);
    std::vector<uint64_t> bitmap;
    std::unique_ptr<faiss::IDGrouperBitmap> idGrouperBitmap = faiss_util::buildIDGrouperBitmap(ids, length, &bitmap);
    int groupIndex = 0;
    for (int i = 0; i <= ids[length - 1]; i++) {
        if (i > ids[groupIndex]) {
            groupIndex++;
        }
        ASSERT_EQ(ids[groupIndex], idGrouperBitmap->get_group(i));
    }
}

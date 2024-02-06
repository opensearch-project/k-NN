/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "knn_extension/faiss/MultiVectorResultCollector.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::NiceMock;
using ::testing::Return;
using idx_t = faiss::idx_t;


TEST(MultiVectorResultCollectorTest, BasicAssertions) {
    // Data
    // Parent ID: 2, ID: 0, Distance: 10
    // Parent ID: 2, ID: 1, Distance: 11
    // Parent ID: 5, ID: 3, Distance: 12
    // Parent ID: 5, ID: 4, Distance: 13
    // After collector handing the data with k = 3, it should return data with id 0 and 2, one from each group.
    // Parent bit set representation: 100100
    int parent_ids[2] = {2, 5};
    FixedBitSet parent_id_filter(parent_ids, 2);

    idx_t ids[] = {0, 1, 2, 3};
    float distances[] = {10, 11, 12, 13};

    os_faiss::MultiVectorResultCollector* rc = new os_faiss::MultiVectorResultCollector(&parent_id_filter, nullptr);
    int k = 3;
    int nres = 0;
    float* bh_val = new float[k];
    int64_t* bh_ids = new int64_t[k];
    for (int i = 0; i < 4; i++) {
        rc->collect(k, nres, bh_val, bh_ids, distances[i], ids[i]);
    }

    // Parent ID is stored before finalize
    ASSERT_EQ(5, bh_ids[0]);
    ASSERT_EQ(2, bh_ids[1]);

    rc->post_process(nres, bh_ids);

    // Parent ID is converted to ID after finalize
    ASSERT_EQ(3, bh_ids[0]);
    ASSERT_EQ(0, bh_ids[1]);

    delete rc;
    delete[] bh_val;
    delete[] bh_ids;
}

TEST(MultiVectorResultCollectorWithIDMapTest, BasicAssertions) {
    // Data
    // Parent ID: 2, Lucene ID: 0, Faiss ID: 0, Distance: 10
    // Parent ID: 2, Lucene ID: 1, Faiss ID: 1, Distance: 11
    // Parent ID: 5, Lucene ID: 3, Faiss ID: 2, Distance: 12
    // Parent ID: 5, Lucene ID: 4, Faiss ID: 3, Distance: 13
    // After collector handing the data with k = 3, it should return data with id 0 and 2, one from each group.

    // Parent bit set representation with Lucene ID: 100100
    int parent_ids[2] = {2, 5};
    FixedBitSet parent_id_filter(parent_ids, 2);

    idx_t faiss_ids[] = {0, 1, 2, 3};
    float distances[] = {10, 11, 12, 13};

    // Faiss IDs to Lucene ID mapping
    std::vector<int64_t> id_map = {0, 1, 3, 4};

    os_faiss::MultiVectorResultCollector* rc = new os_faiss::MultiVectorResultCollector(&parent_id_filter, &id_map);
    int k = 3;
    int nres = 0;
    float* bh_val = new float[k];
    int64_t* bh_ids = new int64_t[k];
    for (int i = 0; i < 4; i++) {
        rc->collect(k, nres, bh_val, bh_ids, distances[i], faiss_ids[i]);
    }

    // Parent ID is stored before finalize
    ASSERT_EQ(5, bh_ids[0]);
    ASSERT_EQ(2, bh_ids[1]);

    rc->post_process(nres, bh_ids);

    // Parent ID is converted to Faiss ID after finalize
    ASSERT_EQ(2, bh_ids[0]);
    ASSERT_EQ(0, bh_ids[1]);

    delete rc;
    delete[] bh_val;
    delete[] bh_ids;
}

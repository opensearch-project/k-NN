/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "knn_extension/faiss/MultiVectorResultCollectorFactory.h"
#include "knn_extension/faiss/MultiVectorResultCollector.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "jni_util.h"

using ::testing::NiceMock;
using ::testing::Return;
using idx_t = faiss::idx_t;


TEST(MultiVectorResultCollectorFactoryTest, BasicAssertions) {
    int parent_ids[1] = {1};
    FixedBitSet parent_id_filter(parent_ids, 1);

    std::unordered_map<idx_t, idx_t> distance1;
    distance1[0] = 10;
    distance1[1] = 11;

    std::unordered_map<idx_t, idx_t> distance2;
    distance2[0] = 11;
    distance2[1] = 10;

    os_faiss::MultiVectorResultCollectorFactory* rc_factory = new os_faiss::MultiVectorResultCollectorFactory(&parent_id_filter);
    faiss::ResultCollector* rc1 = rc_factory->new_collector();
    faiss::ResultCollector* rc2 = rc_factory->new_collector();
    ASSERT_NE(rc1, rc2);

    int k = 1;
    int nres1 = 0;
    int nres2 = 0;
    float* bh_val = new float[k * 2];
    int64_t* bh_ids = new int64_t[k * 2];
    // Verify two collector are thread safe each other.
    // Simulate multi thread by interleaving collect methods of two ResultCollectors.
    for (int i = 0; i < distance1.size(); i++) {
        rc1->collect(k, nres1, bh_val, bh_ids, distance1.at(i), i);
        rc2->collect(k, nres2, bh_val + k, bh_ids + k, distance2.at(i), i);
    }
    rc1->post_process(nres1, bh_ids);
    rc2->post_process(nres2, bh_ids + k);

    ASSERT_EQ(0, bh_ids[0]);
    ASSERT_EQ(1, bh_ids[1]);

    rc_factory->delete_collector(rc1);
    rc_factory->delete_collector(rc2);
    delete rc_factory;
    delete[] bh_val;
    delete[] bh_ids;
}

// Verify that id_map is passed to collector
TEST(MultiVectorResultCollectorFactoryWithIdMapTest, BasicAssertions) {
    int parent_ids[1] = {1};
    FixedBitSet parent_id_filter(parent_ids, 1);
    std::vector<int64_t> id_map;

    os_faiss::MultiVectorResultCollectorFactory* rc_factory = new os_faiss::MultiVectorResultCollectorFactory(&parent_id_filter);
    os_faiss::MultiVectorResultCollector* rc1 = dynamic_cast<os_faiss::MultiVectorResultCollector *>(rc_factory->new_collector());
    ASSERT_EQ(nullptr, rc1->id_map);

    rc_factory->id_map = &id_map;
    os_faiss::MultiVectorResultCollector* rc2 = dynamic_cast<os_faiss::MultiVectorResultCollector *>(rc_factory->new_collector());
    ASSERT_EQ(&id_map, rc2->id_map);

    rc_factory->delete_collector(rc1);
    rc_factory->delete_collector(rc2);
    delete rc_factory;
}

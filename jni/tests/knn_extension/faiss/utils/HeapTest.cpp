/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "knn_extension/faiss/utils/Heap.h"
#include "faiss/utils/Heap.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ElementsAreArray;

TEST(MaxHeapUpdateTest, BasicAssertions) {
    const int k = 5;
    int nres = 0;
    float binary_heap_values[k];
    int64_t binary_heap_ids[k];
    float input_values[] = {1.1f, 2.1f, 3.1f, 4.1f, 5.1f};
    int64_t input_ids[] = {1, 2, 3, 4, 5};
    int64_t group_ids[] = {11, 22, 33, 44, 55};
    std::unordered_map<int64_t, int64_t> group_id_to_id;
    std::unordered_map<int64_t, size_t> group_id_to_index;

    // Push
    for (int i = 0; i < k; i++) {
        os_faiss::maxheap_push(
            nres++,
            binary_heap_values,
            binary_heap_ids,
            input_values[i],
            input_ids[i],
            &group_id_to_id,
            &group_id_to_index,
            group_ids[i]);
    }

    // Verify heap data
    // The top node in the max heap should be the one with max value(5.1f)
    ASSERT_EQ(5.1f, binary_heap_values[0]);
    ASSERT_EQ(55, binary_heap_ids[0]);
    ASSERT_EQ(5, group_id_to_id.at(binary_heap_ids[0]));

    // Replace top
    os_faiss::maxheap_replace_top(
            nres,
            binary_heap_values,
            binary_heap_ids,
            0.1f,
            6,
            &group_id_to_id,
            &group_id_to_index,
            66);

    // Verify heap data
    // Previous top value(5.1f) should have been removed and the next max value(4.1f) should be in the top node.
    ASSERT_EQ(4.1f, binary_heap_values[0]);
    ASSERT_EQ(44, binary_heap_ids[0]);
    ASSERT_EQ(4, group_id_to_id.at(binary_heap_ids[0]));

    // Update
    os_faiss::maxheap_update(
            nres,
            binary_heap_values,
            binary_heap_ids,
            0.2f,
            7,
            &group_id_to_id,
            &group_id_to_index,
            33);

    // Verify heap data
    // node id 3 with group id 33 should have been replaced by node id 7 with new value
    ASSERT_EQ(7, group_id_to_id.at(33));

    // Verify heap is in order
    float expectedValues[] = {4.1f, 2.1f, 1.1f, 0.2f, 0.1f};
    int64_t expectedIds[] = {4, 2, 1, 7, 6};
    for (int i = 0; i < k; i++) {
        ASSERT_EQ(expectedValues[i], binary_heap_values[0]);
        ASSERT_EQ(expectedIds[i], group_id_to_id.at(binary_heap_ids[0]));
        faiss::maxheap_pop(nres--, binary_heap_values, binary_heap_ids);
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <faiss/impl/ResultCollector.h>
#include <faiss/MetricType.h>
#include "knn_extension/faiss/utils/BitSet.h"
#include <unordered_map>

namespace os_faiss {

using idx_t = faiss::idx_t;
/**
 * Implementation of ResultCollector to support multi vector
 *
 * Only supports HNSW algorithm
 *
 * Example:
 * When there is two lucene document with two nested fields, the parent_bit_set value of 100100 is provided where
 * parent doc ids are 2, and 5. Doc id for nested fields of parent document 2 are 0, and 1. Doc id for nested fields
 * of parent document 5 are 3, and 4. For faiss, only nested fields are stored. Therefore corresponding doc ids for
 * nested fields 0, 1, 3, 4 is 0, 1, 2, 3 in faiss. This mapping data is stored in id_map parameter.
 *
 * When collect method is called
 * 1. It switches from faiss id to lucene id and look for its parent id.
 * 2. See if the parent id already exist in heap using either parent_id_to_id or parent_id_to_index.
 * 3. If it does not exist, add the parent id and distance value in the heap(bh_ids, bh_val) and update parent_id_to_id, and parent_id_to_index.
 * 4. If it does exist, update the distance value(bh_val), parent_id_to_id, and parent_id_to_index.
 *
 * When post_process method is called
 * 1. Convert lucene parent ID to faiss doc ID using parent_id_to_id
 */
struct MultiVectorResultCollector:faiss::ResultCollector {
    // BitSet of lucene parent doc ID
    const BitSet* parent_bit_set;

    // Mapping data from Faiss doc ID to Lucene doc ID
    const std::vector<int64_t>* id_map;

    // Lucene parent doc ID to to Faiss doc ID
    // Lucene parent doc ID to index in heap(bh_val, bh_ids)
    std::unordered_map<idx_t, idx_t> parent_id_to_id;
    std::unordered_map<idx_t, size_t> parent_id_to_index;
    MultiVectorResultCollector(const BitSet* parent_bit_set, const std::vector<int64_t>* id_map);

    /**
     *
     * @param k         max size of bh_val, and bh_ids
     * @param nres      number of results in bh_val, and bh_ids
     * @param bh_val    binary heap storing values (For this case distance from query to result)
     * @param bh_ids    binary heap storing document IDs
     * @param val       a new value to add in bh_val
     * @param ids       a new doc id to add in bh_ids
     */
    void collect(
            int k,
            int& nres,
            float* bh_val,
            int64_t* bh_ids,
            float val,
            int64_t ids) override;
    void post_process(int64_t nres, int64_t* bh_ids) override;
};

} // namespace os_faiss


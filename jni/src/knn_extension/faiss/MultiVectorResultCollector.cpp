/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "MultiVectorResultCollector.h"
#include "knn_extension/faiss/utils/Heap.h"
#include "knn_extension/faiss/utils/BitSet.h"

namespace os_faiss {

using idx_t = faiss::idx_t;

MultiVectorResultCollector::MultiVectorResultCollector(const BitSet* parent_bit_set, const std::vector<int64_t>* id_map)
: parent_bit_set(parent_bit_set), id_map(id_map) {}

void MultiVectorResultCollector::collect(
        int k,
        int& nres,
        float* bh_val,
        int64_t* bh_ids,
        float val,
        int64_t ids) {
    idx_t group_id = id_map ? parent_bit_set->next_set_bit(id_map->at(ids)) : parent_bit_set->next_set_bit(ids);
    if (parent_id_to_index.find(group_id) ==
        parent_id_to_index.end()) {
        if (nres < k) {
            maxheap_push(
                    nres++,
                    bh_val,
                    bh_ids,
                    val,
                    ids,
                    &parent_id_to_id,
                    &parent_id_to_index,
                    group_id);
        } else if (val < bh_val[0]) {
            maxheap_replace_top(
                    nres,
                    bh_val,
                    bh_ids,
                    val,
                    ids,
                    &parent_id_to_id,
                    &parent_id_to_index,
                    group_id);
        }
    } else if (val < bh_val[parent_id_to_index.at(group_id)]) {
        maxheap_update(
                nres,
                bh_val,
                bh_ids,
                val,
                ids,
                &parent_id_to_id,
                &parent_id_to_index,
                group_id);
    }
}

void MultiVectorResultCollector::post_process(int64_t nres, int64_t* bh_ids) {
    for (size_t icnt = 0; icnt < nres; icnt++) {
        bh_ids[icnt] = parent_id_to_id.at(bh_ids[icnt]);
    }
}

} // namespace os_faiss

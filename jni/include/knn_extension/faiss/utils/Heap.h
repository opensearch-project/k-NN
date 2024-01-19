/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <climits>
#include <cmath>
#include <cstring>

#include <stdint.h>
#include <cassert>
#include <cstdio>

#include <limits>
#include <unordered_map>
#include <faiss/utils/ordered_key_value.h>

// Collection of heap operations with parent id to dedupe
namespace os_faiss {

/**
 * From start_index, it compare its value with parent node's and swap if needed.
 * Continue until either there is no swap or it reaches the top node.
 *
 * @param bh_val        binary heap storing values
 * @param bh_ids        binary heap storing parent ids
 * @param val           new value to add
 * @param id            new id to add
 * @parent_id_to_id     parent doc id to id mapping data, see MultiVectorResultCollector.h
 * @parent_id_to_index  parent doc id to index mapping data, see MultiVectorResultCollector.h
 * @parent_id           parent id of given id
 * @start_index         an index to start up-heap from in the binary heap(bh_val, and bh_ids)
 */
template <class C>
static inline void up_heap(
        typename C::T* bh_val,
        typename C::TI* bh_ids,
        typename C::T val,
        typename C::TI id,
        std::unordered_map<typename C::TI, typename C::TI>* parent_id_to_id,
        std::unordered_map<typename C::TI, size_t>* parent_id_to_index,
        typename C::TI parent_id,
        size_t start_index) {
    bh_val--; /* Use 1-based indexing for easier node->child translation */
    bh_ids--;
    size_t i = start_index + 1, i_father;

    while (i > 1) {
        i_father = i >> 1;
        if (!C::cmp2(val, bh_val[i_father], parent_id, bh_ids[i_father])) {
            /* the heap structure is ok */
            break;
        }
        bh_val[i] = bh_val[i_father];
        bh_ids[i] = bh_ids[i_father];
        (*parent_id_to_index)[bh_ids[i]] = i - 1;
        i = i_father;
    }
    bh_val[i] = val;
    bh_ids[i] = parent_id;
    (*parent_id_to_id)[parent_id] = id;
    (*parent_id_to_index)[parent_id] = i - 1;
}

/**
 * From start_index, it compare its value with child node's and swap if needed.
 * Continue until either there is no swap or it reaches the leaf node.
 *
 * @param nres          number of values in the binary heap(bh_val, and bh_ids)
 * @param bh_val        binary heap storing values
 * @param bh_ids        binary heap storing parent ids
 * @param val           new value to add
 * @param id            new id to add
 * @parent_id_to_id     parent doc id to id mapping data, see MultiVectorResultCollector.h
 * @parent_id_to_index  parent doc id to index mapping data, see MultiVectorResultCollector.h
 * @parent_id           parent id of given id
 * @start_index         an index to start up-heap from in the binary heap(bh_val, and bh_ids)
 */
template <class C>
static inline void down_heap(
        int nres,
        typename C::T* bh_val,
        typename C::TI* bh_ids,
        typename C::T val,
        typename C::TI id,
        std::unordered_map<typename C::TI, typename C::TI>* parent_id_to_id,
        std::unordered_map<typename C::TI, size_t>* parent_id_to_index,
        typename C::TI parent_id,
        size_t start_index) {
    bh_val--; /* Use 1-based indexing for easier node->child translation */
    bh_ids--;
    size_t i = start_index + 1, i1, i2;

    while (1) {
        i1 = i << 1;
        i2 = i1 + 1;
        if (i1 > nres) {
            break;
        }

        // Note that C::cmp2() is a bool function answering
        // `(a1 > b1) || ((a1 == b1) && (a2 > b2))` for max
        // heap and same with the `<` sign for min heap.
        if ((i2 == nres + 1) ||
            C::cmp2(bh_val[i1], bh_val[i2], bh_ids[i1], bh_ids[i2])) {
            if (C::cmp2(val, bh_val[i1], parent_id, bh_ids[i1])) {
                break;
            }
            bh_val[i] = bh_val[i1];
            bh_ids[i] = bh_ids[i1];
            (*parent_id_to_index)[bh_ids[i]] = i - 1;
            i = i1;
        } else {
            if (C::cmp2(val, bh_val[i2], parent_id, bh_ids[i2])) {
                break;
            }
            bh_val[i] = bh_val[i2];
            bh_ids[i] = bh_ids[i2];
            (*parent_id_to_index)[bh_ids[i]] = i - 1;
            i = i2;
        }
    }
    bh_val[i] = val;
    bh_ids[i] = parent_id;
    (*parent_id_to_id)[parent_id] = id;
    (*parent_id_to_index)[parent_id] = i - 1;
}

/**
 * Push the value to the max heap
 * As the heap contains only one value per group id, pushing a value of existing group id
 * will break the data integrity. For existing group id, use maxheap_update instead.
 * The parent_id should not exist in in bh_ids, parent_id_to_id, and parent_id_to_index.
 *
 * @param nres          number of values in the binary heap(bh_val, and bh_ids)
 * @param bh_val        binary heap storing values
 * @param bh_ids        binary heap storing parent ids
 * @param val           new value to add
 * @param id            new id to add
 * @parent_id_to_id     parent doc id to id mapping data, see MultiVectorResultCollector.h
 * @parent_id_to_index  parent doc id to index mapping data, see MultiVectorResultCollector.h
 * @parent_id           parent id of given id
 */
template <typename T>
inline void maxheap_push(
        int nres,
        T* bh_val,
        int64_t* bh_ids,
        T val,
        int64_t id,
        std::unordered_map<int64_t, int64_t>* parent_id_to_id,
        std::unordered_map<int64_t, size_t>* parent_id_to_index,
        int64_t parent_id) {

    assert(parent_id_to_index->find(parent_id) == parent_id_to_index->end() && "parent id should not exist in the binary heap");

    up_heap<faiss::CMax<T, int64_t>>(
            bh_val,
            bh_ids,
            val,
            id,
            parent_id_to_id,
            parent_id_to_index,
            parent_id,
            nres);
}

/**
 * Update the top node with given value
 * The parent_id should not exist in in bh_ids, parent_id_to_id, and parent_id_to_index.
 *
 * @param nres          number of values in the binary heap(bh_val, and bh_ids)
 * @param bh_val        binary heap storing values
 * @param bh_ids        binary heap storing parent ids
 * @param val           new value to add
 * @param id            new id to add
 * @parent_id_to_id     parent doc id to id mapping data, see MultiVectorResultCollector.h
 * @parent_id_to_index  parent doc id to index mapping data, see MultiVectorResultCollector.h
 * @parent_id           parent id of given id
 */
template <typename T>
inline void maxheap_replace_top(
        int nres,
        T* bh_val,
        int64_t* bh_ids,
        T val,
        int64_t id,
        std::unordered_map<int64_t, int64_t>* parent_id_to_id,
        std::unordered_map<int64_t, size_t>* parent_id_to_index,
        int64_t parent_id) {

    assert(parent_id_to_index->find(parent_id) == parent_id_to_index->end() && "parent id should not exist in the binary heap");

    parent_id_to_id->erase(bh_ids[0]);
    parent_id_to_index->erase(bh_ids[0]);
    down_heap<faiss::CMax<T, int64_t>>(
            nres,
            bh_val,
            bh_ids,
            val,
            id,
            parent_id_to_id,
            parent_id_to_index,
            parent_id,
            0);
}

/**
 * Update value of the parent_id in the binary heap and id of the parent_id in parent_id_to_id
 * The parent_id should exist in bh_ids, parent_id_to_id, and parent_id_to_index.
 *
 * @param nres          number of values in the binary heap(bh_val, and bh_ids)
 * @param bh_val        binary heap storing values
 * @param bh_ids        binary heap storing parent ids
 * @param val           new value to update
 * @param id            new id to update
 * @parent_id_to_id     parent doc id to id mapping data, see MultiVectorResultCollector.h
 * @parent_id_to_index  parent doc id to index mapping data, see MultiVectorResultCollector.h
 * @parent_id           parent id of given id
 */
template <typename T>
inline void maxheap_update(
        int nres,
        T* bh_val,
        int64_t* bh_ids,
        T val,
        int64_t id,
        std::unordered_map<int64_t, int64_t>* parent_id_to_id,
        std::unordered_map<int64_t, size_t>* parent_id_to_index,
        int64_t parent_id) {
        size_t target_index = parent_id_to_index->at(parent_id);
    up_heap<faiss::CMax<T, int64_t>>(
            bh_val,
            bh_ids,
            val,
            id,
            parent_id_to_id,
            parent_id_to_index,
            parent_id,
            target_index);
    down_heap<faiss::CMax<T, int64_t>>(
            nres,
            bh_val,
            bh_ids,
            val,
            id,
            parent_id_to_id,
            parent_id_to_index,
            parent_id,
            target_index);
}

}  // namespace os_faiss

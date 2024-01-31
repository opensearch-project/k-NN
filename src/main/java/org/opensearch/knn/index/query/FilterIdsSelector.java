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

package org.opensearch.knn.index.query;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.FixedBitSet;

import java.io.IOException;

import static org.opensearch.knn.common.KNNConstants.MAX_ID_SELECT_ARRAY;

/**
 * Util Class for filter ids selector
 */
public class FilterIdsSelector {

    /**
     * When do ann query with filters, there are two types:
     * BitMap using FixedBitSet, BATCH using a long array stands for filter result docids.
     */
    public enum FilterIdsSelectorType {

        BITMAP(0),

        BATCH(1);

        private final int value;

        FilterIdsSelectorType(int value) {
            this.value = value;
        }

        public int getValue() {
            return value;
        }
    }

    long[] filterIds;
    FilterIdsSelectorType filterType;

    public FilterIdsSelector(long[] filterIds, FilterIdsSelectorType type) {
        this.filterIds = filterIds;
        this.filterType = type;
    }

    public long[] getFilterIds() {
        return filterIds;
    }

    public FilterIdsSelectorType getFilterType() {
        return filterType;
    }

    /**
     * This function takes a call on what ID Selector to use:
     * https://github.com/facebookresearch/faiss/wiki/Setting-search-parameters-for-one-query#idselectorarray-idselectorbatch-and-idselectorbitmap
     *
     * class	       storage	lookup     construction(Opensearch + Faiss)
     * IDSelectorArray	O(k)	O(k)          O(2k)
     * IDSelectorBatch	O(k)	O(1)          O(2k)
     * IDSelectorBitmap	O(n/8)	O(1)          O(k) -> n is the max value of id in the index
     *
     * The Goal selector is to keep the size balance with bitmap and array.
     * keep the memory and lookup at an upper limit usage.
     * So MAX_ID_SELECT_ARRAY keep less than 2M docids which less than 15MB memory.
     * When using FIXEDBitSet can store almost 0.1B docids with 15MB memory.
     *
     * @param filterIdsBitSet
     * @param cardinality
     * @return Tuple<long[], FilterIdsSelector.FilterIdsSelectorType>
     */
    public static FilterIdsSelector getIdSelectorType(BitSet filterIdsBitSet, int cardinality) throws IOException {
        long[] filterIds;
        FilterIdsSelector.FilterIdsSelectorType filterType;
        if (filterIdsBitSet instanceof FixedBitSet) {
            /**
             * When filterIds is dense filter, using fixed bitset
             */
            filterIds = ((FixedBitSet) filterIdsBitSet).getBits();
            filterType = FilterIdsSelector.FilterIdsSelectorType.BITMAP;
        } else if (cardinality < MAX_ID_SELECT_ARRAY) {
            /**
             * When filterIds is Sparse filter, using BATCH filter.
             */
            BitSetIterator bitSetIterator = new BitSetIterator(filterIdsBitSet, cardinality);
            filterIds = new long[cardinality];
            int idx = 0;
            for (int docId = bitSetIterator.nextDoc(); docId != DocIdSetIterator.NO_MORE_DOCS; docId = bitSetIterator.nextDoc()) {
                filterIds[idx++] = docId;
            }
            filterType = FilterIdsSelector.FilterIdsSelectorType.BATCH;
        } else {
            /**
             * Others using fixed bitset, may be SparseBitSet
             */
            int length = filterIdsBitSet.length();
            FixedBitSet fixedBitSet = new FixedBitSet(length);
            BitSetIterator bitSetIterator = new BitSetIterator(filterIdsBitSet, cardinality);
            fixedBitSet.or(bitSetIterator);
            filterIds = fixedBitSet.getBits();
            filterType = FilterIdsSelector.FilterIdsSelectorType.BITMAP;
        }
        return new FilterIdsSelector(filterIds, filterType);
    }
}

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

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.FixedBitSet;

import java.io.IOException;

/**
 * Util Class for filter ids selector
 */
@AllArgsConstructor
@Getter
public class FilterIdsSelector {

    /**
     * When do ann query with filters, there are two types:
     * BitMap using FixedBitSet, BATCH using a long array stands for filter result docids.
     */
    @AllArgsConstructor
    @Getter
    public enum FilterIdsSelectorType {
        BITMAP(0),
        BATCH(1);

        private final int value;
    }

    long[] filterIds;
    private FilterIdsSelectorType filterType;

    /**
     * This function takes a call on what ID Selector to use:
     * https://github.com/facebookresearch/faiss/wiki/Setting-search-parameters-for-one-query#idselectorarray-idselectorbatch-and-idselectorbitmap
     *
     * class	       storage	lookup     construction(Opensearch + Faiss)
     * IDSelectorArray	O(k)	O(k)          O(2k)
     * IDSelectorBatch	O(k)	O(1)          O(2k)
     * IDSelectorBitmap	O(n/8)	O(1)          O(k) n is the max value of id in the index
     *
     * TODO: We need to ideally decide when we can take another hit of K iterations in latency. Some facts:
     * an OpenSearch Index can have max segment size as 5GB which, which on a vector with dimension of 128 boils down to
     * 7.5M vectors.
     * Ref: https://opensearch.org/docs/latest/search-plugins/knn/knn-index/#hnsw-memory-estimation
     * M = 16
     * Dimension = 128
     * (1.1 * ( 4 * 128 + 8 * 16) * 7500000)/(1024*1024*1024) ~ 4.9GB
     * Ids are sequential in a Segment which means for IDSelectorBitmap total size if the max ID has value of 7.5M will be
     * 7500000/(8*1024) = 915KBs in worst case. But with larger dimensions this worst case value will decrease.
     *
     * With 915KB how many ids can be represented as an array of 64-bit longs : 117,120 ids
     * So iterating on 117k ids for 1 single pass is also time consuming. So, we are currently concluding to consider only size
     * as factor. We need to improve on this.
     *
     * Array Memory: Cardinality * Long.BYTES
     * BitSet Memory: MaxId / Byte.SIZE
     * When Array Memory less than or equal to BitSet Memory return FilterIdsSelectorType.BATCH
     * Else return FilterIdsSelectorType.BITMAP;
     *
     * @param filterIdsBitSet Filter query result docs
     * @param cardinality The number of bits that are set
     * @return {@link FilterIdsSelector}
     */
    public static FilterIdsSelector getFilterIdSelector(final BitSet filterIdsBitSet, final int cardinality) throws IOException {
        long[] filterIds;
        FilterIdsSelector.FilterIdsSelectorType filterType;
        if (filterIdsBitSet == null) {
            filterIds = null;
            filterType = FilterIdsSelector.FilterIdsSelectorType.BITMAP;
        } else if (filterIdsBitSet instanceof FixedBitSet) {
            /**
             * When filterIds is dense filter, using fixed bitset
             */
            filterIds = ((FixedBitSet) filterIdsBitSet).getBits();
            filterType = FilterIdsSelector.FilterIdsSelectorType.BITMAP;
        } else if ((cardinality * Long.BYTES * Byte.SIZE) <= filterIdsBitSet.length()) {
            /**
             * When filterIds is sparse bitset, using ram usage to decide FilterIdsSelectorType
             */
            BitSetIterator bitSetIterator = new BitSetIterator(filterIdsBitSet, cardinality);
            filterIds = new long[cardinality];
            int idx = 0;
            for (int docId = bitSetIterator.nextDoc(); docId != DocIdSetIterator.NO_MORE_DOCS; docId = bitSetIterator.nextDoc()) {
                filterIds[idx++] = docId;
            }
            filterType = FilterIdsSelectorType.BATCH;
        } else {
            FixedBitSet fixedBitSet = new FixedBitSet(filterIdsBitSet.length());
            BitSetIterator sparseBitSetIterator = new BitSetIterator(filterIdsBitSet, cardinality);
            fixedBitSet.or(sparseBitSetIterator);
            filterIds = fixedBitSet.getBits();
            filterType = FilterIdsSelector.FilterIdsSelectorType.BITMAP;
        }
        return new FilterIdsSelector(filterIds, filterType);
    }
}

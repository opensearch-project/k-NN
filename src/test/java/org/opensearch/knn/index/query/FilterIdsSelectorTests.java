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

import lombok.SneakyThrows;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.SparseFixedBitSet;
import org.opensearch.knn.KNNTestCase;

public class FilterIdsSelectorTests extends KNNTestCase {

    @SneakyThrows
    public void testGetIdSelectorTypeWithFixedBitSet() {
        FixedBitSet bits = new FixedBitSet(101);
        for (int i = 1; i <= 100; i++) {
            bits.set(i);
        }
        FilterIdsSelector idsSelector = FilterIdsSelector.getFilterIdSelector(bits, bits.cardinality());
        assertEquals(idsSelector.getFilterType(), FilterIdsSelector.FilterIdsSelectorType.BITMAP);
        assertArrayEquals(bits.getBits(), idsSelector.filterIds);
    }

    @SneakyThrows
    public void testGetIdSelectorTypeWithSparseBitSetHigh() {
        SparseFixedBitSet bits = new SparseFixedBitSet(101);
        for (int i = 1; i <= 100; i++) {
            bits.set(i);
        }
        FilterIdsSelector idsSelector = FilterIdsSelector.getFilterIdSelector(bits, bits.cardinality());
        assertEquals(idsSelector.getFilterType(), FilterIdsSelector.FilterIdsSelectorType.BITMAP);
        FixedBitSet fixedBitSet = new FixedBitSet(bits.length());
        BitSetIterator sparseBitSetIterator = new BitSetIterator(bits, 101);
        fixedBitSet.or(sparseBitSetIterator);
        assertArrayEquals(fixedBitSet.getBits(), idsSelector.filterIds);
    }

    @SneakyThrows
    public void testGetIdSelectorTypeWithSparseBitSetLow() {
        int maxDoc = (Integer.MAX_VALUE) / 2;
        SparseFixedBitSet bits = new SparseFixedBitSet(maxDoc);
        long array[] = new long[100];
        for (int i = maxDoc - 100, idx = 0; i < maxDoc; i++) {
            bits.set(i);
            array[idx++] = i;
        }
        FilterIdsSelector idsSelector = FilterIdsSelector.getFilterIdSelector(bits, bits.cardinality());
        assertEquals(idsSelector.getFilterType(), FilterIdsSelector.FilterIdsSelectorType.BATCH);
        assertArrayEquals(array, idsSelector.filterIds);
    }
}

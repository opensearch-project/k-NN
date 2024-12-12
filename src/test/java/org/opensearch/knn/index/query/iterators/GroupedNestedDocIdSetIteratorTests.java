/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.iterators;

import junit.framework.TestCase;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.FixedBitSet;

import java.util.HashSet;
import java.util.Set;

public class GroupedNestedDocIdSetIteratorTests extends TestCase {
    public void testGroupedNestedDocIdSetIterator_whenNextDocIsCalled_thenBehaveAsExpected() throws Exception {
        // 0, 1, 2(parent), 3, 4, 5, 6, 7(parent), 8, 9, 10(parent)
        BitSet parentBitSet = new FixedBitSet(new long[1], 11);
        parentBitSet.set(2);
        parentBitSet.set(7);
        parentBitSet.set(10);

        BitSet filterBits = new FixedBitSet(new long[1], 11);
        filterBits.set(1);
        filterBits.set(8);
        filterBits.set(9);

        // Run
        Set<Integer> docIds = Set.of(1, 8);
        GroupedNestedDocIdSetIterator groupedNestedDocIdSetIterator = new GroupedNestedDocIdSetIterator(parentBitSet, docIds, filterBits);

        // Verify
        Set<Integer> expectedDocIds = Set.of(1, 8, 9);
        Set<Integer> returnedDocIds = new HashSet<>();
        groupedNestedDocIdSetIterator.nextDoc();
        while (groupedNestedDocIdSetIterator.docID() != DocIdSetIterator.NO_MORE_DOCS) {
            returnedDocIds.add(groupedNestedDocIdSetIterator.docID());
            groupedNestedDocIdSetIterator.nextDoc();
        }
        assertEquals(expectedDocIds, returnedDocIds);
        assertEquals(expectedDocIds.size(), groupedNestedDocIdSetIterator.cost());
    }

    public void testGroupedNestedDocIdSetIterator_whenAdvanceIsCalled_thenBehaveAsExpected() throws Exception {
        // 0, 1, 2(parent), 3, 4, 5, 6, 7(parent), 8, 9, 10(parent)
        BitSet parentBitSet = new FixedBitSet(new long[1], 11);
        parentBitSet.set(2);
        parentBitSet.set(7);
        parentBitSet.set(10);

        BitSet filterBits = new FixedBitSet(new long[1], 11);
        filterBits.set(1);
        filterBits.set(8);
        filterBits.set(9);

        // Run
        Set<Integer> docIds = Set.of(1, 8);
        GroupedNestedDocIdSetIterator groupedNestedDocIdSetIterator = new GroupedNestedDocIdSetIterator(parentBitSet, docIds, filterBits);

        // Verify
        Set<Integer> expectedDocIds = Set.of(1, 8, 9);
        groupedNestedDocIdSetIterator.advance(1);
        assertEquals(1, groupedNestedDocIdSetIterator.docID());
        groupedNestedDocIdSetIterator.advance(8);
        assertEquals(8, groupedNestedDocIdSetIterator.docID());
        groupedNestedDocIdSetIterator.advance(9);
        assertEquals(9, groupedNestedDocIdSetIterator.docID());
        groupedNestedDocIdSetIterator.nextDoc();
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, groupedNestedDocIdSetIterator.docID());
        assertEquals(expectedDocIds.size(), groupedNestedDocIdSetIterator.cost());
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.exactsearch.scorers;

import junit.framework.TestCase;
import lombok.SneakyThrows;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.FixedBitSet;

import java.io.IOException;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class NestedBestChildVectorScorerTests extends TestCase {

    @SneakyThrows
    public void testIterator_whenCalled_returnsSameInstance() {
        VectorScorer mockVectorScorer = mock(VectorScorer.class);
        DocIdSetIterator mockVectorIterator = mock(DocIdSetIterator.class);
        when(mockVectorScorer.iterator()).thenReturn(mockVectorIterator);

        FixedBitSet childrenBitSet = new FixedBitSet(4);
        childrenBitSet.set(0);
        DocIdSetIterator acceptedChildrenIterator = new BitSetIterator(childrenBitSet, childrenBitSet.length());

        BitSet parentBitSet = new FixedBitSet(new long[] { 2 }, 2);

        NestedBestChildVectorScorer scorer = new NestedBestChildVectorScorer(acceptedChildrenIterator, parentBitSet, mockVectorScorer);

        DocIdSetIterator iterator1 = scorer.iterator();
        DocIdSetIterator iterator2 = scorer.iterator();
        assertSame("Iterator should return same instance", iterator1, iterator2);
    }

    @SneakyThrows
    public void testNextParent_withoutFilter_returnsBestChildPerParent() throws IOException {
        // Setup mock VectorScorer with scores for all children
        VectorScorer mockVectorScorer = mock(VectorScorer.class);
        DocIdSetIterator mockVectorIterator = mock(DocIdSetIterator.class);
        when(mockVectorScorer.iterator()).thenReturn(mockVectorIterator);

        // Mock the iterator behavior for unfiltered case
        // docID() starts at -1, then becomes 0, 2, 3 as nextDoc() is called
        when(mockVectorIterator.docID()).thenReturn(-1, 0, 2, 3, DocIdSetIterator.NO_MORE_DOCS);
        when(mockVectorIterator.nextDoc()).thenReturn(0, 2, 3, DocIdSetIterator.NO_MORE_DOCS);

        // Mock scores based on when they're called:
        // First parent: child 0 gets score 0.8
        // Second parent: child 2 gets score 0.6, child 3 gets score 0.9
        when(mockVectorScorer.score()).thenReturn(0.8f, 0.6f, 0.9f);

        // Setup parent bit set (parents at positions 1, 4)
        // Parent id for 0 -> 1, Parent id for 2,3 -> 4
        // Binary: 10010 = 18 decimal
        BitSet parentBitSet = new FixedBitSet(new long[] { 18 }, 5);

        // Create scorer without filter (acceptedChildrenIterator = null)
        NestedBestChildVectorScorer scorer = new NestedBestChildVectorScorer(parentBitSet, mockVectorScorer);

        DocIdSetIterator iterator = scorer.iterator();

        // First parent (id=1) should return child 0 with score 0.8
        assertEquals(0, iterator.nextDoc());
        assertEquals(0.8f, scorer.score(), 0.001f);

        // Second parent (id=4) should return child 3 with score 0.9 (best among 2 and 3)
        assertEquals(3, iterator.nextDoc());
        assertEquals(0.9f, scorer.score(), 0.001f);

        // No more parents
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    @SneakyThrows
    public void testNextParent_withoutFilter_multipleParentsWithVaryingChildren() throws IOException {
        // Layout: children [0,1,2,3,4] -> parent 5, children [6,7,8] -> parent 9, child [10] -> parent 11
        VectorScorer mockVectorScorer = mock(VectorScorer.class);
        DocIdSetIterator mockVectorIterator = mock(DocIdSetIterator.class);
        when(mockVectorScorer.iterator()).thenReturn(mockVectorIterator);

        // docID() sequence: -1 (initial), then returns current position after each nextDoc()
        // nextChildDoc reads docID first; if -1 calls nextDoc, otherwise returns docID directly
        // advanceVectorIterator calls nextDoc to move past current child
        when(mockVectorIterator.docID()).thenReturn(
            -1,  // 1st nextChildDoc: initial state -> triggers nextDoc
            // Parent 5 group
            1,   // 2nd nextChildDoc (while-cond after child 0)
            2,   // 3rd nextChildDoc (while-cond after child 1)
            3,   // 4th nextChildDoc (while-cond after child 2)
            4,   // 5th nextChildDoc (while-cond after child 3)
            6,   // 6th nextChildDoc (while-cond after child 4) -> 6 >= 5, exits loop
            // Parent 9 group
            6,   // 7th nextChildDoc: start of 2nd nextDoc call
            7,   // 8th nextChildDoc (while-cond after child 6)
            8,   // 9th nextChildDoc (while-cond after child 7)
            10,  // 10th nextChildDoc (while-cond after child 8) -> 10 >= 9, exits loop
            // Parent 11 group
            10,  // 11th nextChildDoc: start of 3rd nextDoc call
            DocIdSetIterator.NO_MORE_DOCS, // 12th nextChildDoc (while-cond after child 10)
            // 4th nextDoc call
            DocIdSetIterator.NO_MORE_DOCS  // 13th nextChildDoc -> NO_MORE_DOCS
        );

        // nextDoc() calls: first from nextChildDoc (initial), rest from advanceVectorIterator
        when(mockVectorIterator.nextDoc()).thenReturn(
            0,   // nextChildDoc initial
            1,   // advanceVectorIterator after child 0
            2,   // advanceVectorIterator after child 1
            3,   // advanceVectorIterator after child 2
            4,   // advanceVectorIterator after child 3
            6,   // advanceVectorIterator after child 4 (skips parent 5)
            7,   // advanceVectorIterator after child 6
            8,   // advanceVectorIterator after child 7
            10,  // advanceVectorIterator after child 8 (skips parent 9)
            DocIdSetIterator.NO_MORE_DOCS  // advanceVectorIterator after child 10
        );

        // Scores: children 0-4 for parent 5, children 6-8 for parent 9, child 10 for parent 11
        // Best child for parent 5: child 2 (0.95), parent 9: child 7 (0.85), parent 11: child 10 (0.5)
        when(mockVectorScorer.score()).thenReturn(
            0.3f,
            0.7f,
            0.95f,
            0.4f,
            0.1f,  // children 0,1,2,3,4
            0.6f,
            0.85f,
            0.2f,               // children 6,7,8
            0.5f                              // child 10
        );

        // Parent bits at positions 5, 9, 11 -> binary: 100000100000 but we need bits 5,9,11
        // bit 5 = 32, bit 9 = 512, bit 11 = 2048 -> 32 + 512 + 2048 = 2592
        BitSet parentBitSet = new FixedBitSet(new long[] { 2592 }, 12);

        NestedBestChildVectorScorer scorer = new NestedBestChildVectorScorer(parentBitSet, mockVectorScorer);
        DocIdSetIterator iterator = scorer.iterator();

        // First parent (5): best child is 2 with score 0.95
        assertEquals(2, iterator.nextDoc());
        assertEquals(0.95f, scorer.score(), 0.001f);

        // Second parent (9): best child is 7 with score 0.85
        assertEquals(7, iterator.nextDoc());
        assertEquals(0.85f, scorer.score(), 0.001f);

        // Third parent (11): only child 10 with score 0.5
        assertEquals(10, iterator.nextDoc());
        assertEquals(0.5f, scorer.score(), 0.001f);

        // No more parents
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    @SneakyThrows
    public void testNextParent_withFilter_returnsBestChildPerParent() {
        VectorScorer mockVectorScorer = mock(VectorScorer.class);
        DocIdSetIterator mockVectorIterator = mock(DocIdSetIterator.class);
        when(mockVectorScorer.iterator()).thenReturn(mockVectorIterator);

        when(mockVectorScorer.score()).thenReturn(0.8f, 0.6f, 0.9f);
        when(mockVectorIterator.advance(0)).thenReturn(0);
        when(mockVectorIterator.advance(2)).thenReturn(2);
        when(mockVectorIterator.advance(3)).thenReturn(3);

        // Setup accepted children iterator (children docs: 0, 2, 3)
        FixedBitSet childrenBitSet = new FixedBitSet(4);
        childrenBitSet.set(0);
        childrenBitSet.set(2);
        childrenBitSet.set(3);
        DocIdSetIterator acceptedChildrenIterator = new BitSetIterator(childrenBitSet, childrenBitSet.length());

        // Setup parent bit set (parents at positions 1, 4)
        BitSet parentBitSet = new FixedBitSet(new long[] { 18 }, 5);

        NestedBestChildVectorScorer scorer = new NestedBestChildVectorScorer(acceptedChildrenIterator, parentBitSet, mockVectorScorer);

        DocIdSetIterator iterator = scorer.iterator();

        // First parent (id=1) should return child 0 with score 0.8
        assertEquals(0, iterator.nextDoc());
        assertEquals(0.8f, scorer.score(), 0.001f);

        // Second parent (id=4) should return child 3 with score 0.9 (best among 2 and 3)
        assertEquals(3, iterator.nextDoc());
        assertEquals(0.9f, scorer.score(), 0.001f);

        // No more parents
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    @SneakyThrows
    public void testNextParent_withFilter_multipleParentsWithVaryingChildren() throws IOException {
        // Same layout as unfiltered test: children [0,1,2,3,4] -> parent 5, children [6,7,8] -> parent 9, child [10] -> parent 11
        // Filter excludes best children: child 2 (0.95) and child 7 (0.85)
        // Accepted children: 0, 1, 3, 4, 6, 8, 10
        // Expected second-best: parent 5 -> child 1 (0.7), parent 9 -> child 6 (0.6), parent 11 -> child 10 (0.5)
        VectorScorer mockVectorScorer = mock(VectorScorer.class);
        DocIdSetIterator mockVectorIterator = mock(DocIdSetIterator.class);
        when(mockVectorScorer.iterator()).thenReturn(mockVectorIterator);

        // Scores in order of visited accepted children:
        // Parent 5: child 0 (0.3), child 1 (0.7), child 3 (0.4), child 4 (0.1)
        // Parent 9: child 6 (0.6), child 8 (0.2)
        // Parent 11: child 10 (0.5)
        when(mockVectorScorer.score()).thenReturn(
            0.3f,
            0.7f,
            0.4f,
            0.1f,  // children 0,1,3,4 (child 2 filtered out)
            0.6f,
            0.2f,               // children 6,8 (child 7 filtered out)
            0.5f                       // child 10
        );

        when(mockVectorIterator.advance(0)).thenReturn(0);
        when(mockVectorIterator.advance(1)).thenReturn(1);
        when(mockVectorIterator.advance(3)).thenReturn(3);
        when(mockVectorIterator.advance(4)).thenReturn(4);
        when(mockVectorIterator.advance(6)).thenReturn(6);
        when(mockVectorIterator.advance(8)).thenReturn(8);
        when(mockVectorIterator.advance(10)).thenReturn(10);

        // Accepted children: 0, 1, 3, 4, 6, 8, 10 (excluding 2 and 7)
        FixedBitSet childrenBitSet = new FixedBitSet(11);
        childrenBitSet.set(0);
        childrenBitSet.set(1);
        childrenBitSet.set(3);
        childrenBitSet.set(4);
        childrenBitSet.set(6);
        childrenBitSet.set(8);
        childrenBitSet.set(10);
        DocIdSetIterator acceptedChildrenIterator = new BitSetIterator(childrenBitSet, childrenBitSet.length());

        // Parent bits at positions 5, 9, 11 -> 32 + 512 + 2048 = 2592
        BitSet parentBitSet = new FixedBitSet(new long[] { 2592 }, 12);

        NestedBestChildVectorScorer scorer = new NestedBestChildVectorScorer(acceptedChildrenIterator, parentBitSet, mockVectorScorer);
        DocIdSetIterator iterator = scorer.iterator();

        // First parent (5): best accepted child is 1 with score 0.7 (child 2 was filtered out)
        assertEquals(1, iterator.nextDoc());
        assertEquals(0.7f, scorer.score(), 0.001f);

        // Second parent (9): best accepted child is 6 with score 0.6 (child 7 was filtered out)
        assertEquals(6, iterator.nextDoc());
        assertEquals(0.6f, scorer.score(), 0.001f);

        // Third parent (11): only child 10 with score 0.5
        assertEquals(10, iterator.nextDoc());
        assertEquals(0.5f, scorer.score(), 0.001f);

        // No more parents
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());

        // Verify advance was never called for filtered-out children
        verify(mockVectorIterator, never()).advance(2);
        verify(mockVectorIterator, never()).advance(7);
    }
}

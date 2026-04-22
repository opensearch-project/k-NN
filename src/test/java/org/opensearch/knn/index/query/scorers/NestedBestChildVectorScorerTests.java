/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.scorers;

import junit.framework.TestCase;
import lombok.SneakyThrows;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.FixedBitSet;

import java.io.IOException;
import java.util.Map;

/**
 * Tests for {@link NestedBestChildVectorScorer}.
 * <p>
 * Document layout used across tests:
 *   children [0, 1, 2, 3, 4] → parent 5
 *   children [6, 7, 8]       → parent 9
 *   child    [10]             → parent 11
 * <p>
 * Parent bits at positions 5, 9, 11 → long value = 2592.
 */
public class NestedBestChildVectorScorerTests extends TestCase {

    private static final int NUM_DOCS = 12;
    // Parent bits: positions 5, 9, 11 → 2^5 + 2^9 + 2^11 = 32 + 512 + 2048 = 2592
    private static final BitSet PARENT_BIT_SET = new FixedBitSet(new long[] { 2592 }, NUM_DOCS);

    // Scores per child doc id
    private static final Map<Integer, Float> SCORES = Map.of(
        0,
        0.3f,
        1,
        0.7f,
        2,
        0.95f,
        3,
        0.4f,
        4,
        0.1f,
        6,
        0.6f,
        7,
        0.85f,
        8,
        0.2f,
        10,
        0.5f
    );

    // ──────────────────────────────────────────────
    // Section 1: Without filter
    // ──────────────────────────────────────────────

    @SneakyThrows
    public void testWithoutFilter_iteratorReturnsSameInstance() {
        NestedBestChildVectorScorer scorer = createScorer(null);
        assertSame(scorer.iterator(), scorer.iterator());
    }

    @SneakyThrows
    public void testWithoutFilter_nextDocAndScore() {
        NestedBestChildVectorScorer scorer = createScorer(null);
        DocIdSetIterator iterator = scorer.iterator();

        // Parent 5: best child is 2 (score 0.95)
        assertEquals(2, iterator.nextDoc());
        assertEquals(0.95f, scorer.score(), 0.001f);

        // Parent 9: best child is 7 (score 0.85)
        assertEquals(7, iterator.nextDoc());
        assertEquals(0.85f, scorer.score(), 0.001f);

        // Parent 11: only child 10 (score 0.5)
        assertEquals(10, iterator.nextDoc());
        assertEquals(0.5f, scorer.score(), 0.001f);

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    @SneakyThrows
    public void testWithoutFilter_cost() {
        FixedBitSet vectorBits = vectorBitSet();
        long expectedCost = new BitSetIterator(vectorBits, vectorBits.length()).cost();

        NestedBestChildVectorScorer scorer = createScorer(null);
        assertEquals(expectedCost, scorer.iterator().cost());
    }

    // ──────────────────────────────────────────────
    // Section 2: With filter
    // ──────────────────────────────────────────────

    @SneakyThrows
    public void testWithFilter_iteratorReturnsSameInstance() {
        NestedBestChildVectorScorer scorer = createScorer(filterBitSet());
        assertSame(scorer.iterator(), scorer.iterator());
    }

    @SneakyThrows
    public void testWithFilter_nextDocAndScore() {
        // Filter excludes children 2 and 7 (the unfiltered best children for parents 5 and 9)
        NestedBestChildVectorScorer scorer = createScorer(filterBitSet());
        DocIdSetIterator iterator = scorer.iterator();

        // Parent 5: best accepted child is 1 (score 0.7), child 2 filtered out
        assertEquals(1, iterator.nextDoc());
        assertEquals(0.7f, scorer.score(), 0.001f);

        // Parent 9: best accepted child is 6 (score 0.6), child 7 filtered out
        assertEquals(6, iterator.nextDoc());
        assertEquals(0.6f, scorer.score(), 0.001f);

        // Parent 11: only child 10 (score 0.5)
        assertEquals(10, iterator.nextDoc());
        assertEquals(0.5f, scorer.score(), 0.001f);

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    @SneakyThrows
    public void testWithFilter_cost() {
        FixedBitSet filter = filterBitSet();
        FixedBitSet vectors = vectorBitSet();
        // Conjunction cost is the minimum of the two iterator costs
        long expectedCost = Math.min(
            new BitSetIterator(filter, filter.length()).cost(),
            new BitSetIterator(vectors, vectors.length()).cost()
        );

        NestedBestChildVectorScorer scorer = createScorer(filter);
        assertEquals(expectedCost, scorer.iterator().cost());
    }

    /**
     * Filter accepts doc ids that have no vector. The conjunction iterator should skip
     * those automatically, and if an entire parent group has no children with vectors,
     * that parent group should be skipped entirely.
     *
     * Layout (same parents at 5, 9, 11):
     *   Vectors exist for:  {0, 4, 8, 10}
     *   Filter accepts:     {0, 1, 2, 3, 4, 6, 7, 8, 10}  (children 1,2,3,6,7 have no vector)
     *   Effective children:  {0, 4} → parent 5,  {8} → parent 9,  {10} → parent 11
     */
    @SneakyThrows
    public void testWithFilter_filterIncludesChildrenWithoutVectors() {
        Map<Integer, Float> sparseScores = Map.of(0, 0.3f, 4, 0.9f, 8, 0.6f, 10, 0.5f);

        FixedBitSet vectorBits = new FixedBitSet(NUM_DOCS);
        sparseScores.keySet().forEach(vectorBits::set);
        DocIdSetIterator vectorIter = new BitSetIterator(vectorBits, vectorBits.length());

        VectorScorer baseScorer = new VectorScorer() {
            @Override
            public float score() throws IOException {
                int doc = vectorIter.docID();
                Float s = sparseScores.get(doc);
                if (s == null) {
                    throw new IllegalStateException("No score for doc " + doc);
                }
                return s;
            }

            @Override
            public DocIdSetIterator iterator() {
                return vectorIter;
            }
        };

        // Filter accepts children 0-4, 6-8, 10 (all children except 5/9/11 which are parents)
        FixedBitSet filterBits = new FixedBitSet(NUM_DOCS);
        for (int i : new int[] { 0, 1, 2, 3, 4, 6, 7, 8, 10 }) {
            filterBits.set(i);
        }
        DocIdSetIterator filterIter = new BitSetIterator(filterBits, filterBits.length());

        NestedBestChildVectorScorer scorer = new NestedBestChildVectorScorer(filterIter, PARENT_BIT_SET, baseScorer);
        DocIdSetIterator iterator = scorer.iterator();

        // Parent 5: children 1,2,3 have no vector and are skipped; best of {0,4} is child 4 (0.9)
        assertEquals(4, iterator.nextDoc());
        assertEquals(0.9f, scorer.score(), 0.001f);

        // Parent 9: children 6,7 have no vector and are skipped; only child 8 (0.6)
        assertEquals(8, iterator.nextDoc());
        assertEquals(0.6f, scorer.score(), 0.001f);

        // Parent 11: child 10 (0.5)
        assertEquals(10, iterator.nextDoc());
        assertEquals(0.5f, scorer.score(), 0.001f);

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    /**
     * Filter accepts children that have no vectors, and for one parent group none of the
     * accepted children have vectors. That parent group should be skipped entirely.
     *
     * Layout (same parents at 5, 9, 11):
     *   Vectors exist for:  {0, 10}
     *   Filter accepts:     {0, 6, 7, 8, 10}  (children 6,7,8 have no vector)
     *   Effective children:  {0} → parent 5,  {} → parent 9 (skipped),  {10} → parent 11
     */
    @SneakyThrows
    public void testWithFilter_entireParentGroupSkippedWhenNoChildrenHaveVectors() {
        Map<Integer, Float> sparseScores = Map.of(0, 0.4f, 10, 0.7f);

        FixedBitSet vectorBits = new FixedBitSet(NUM_DOCS);
        sparseScores.keySet().forEach(vectorBits::set);
        DocIdSetIterator vectorIter = new BitSetIterator(vectorBits, vectorBits.length());

        VectorScorer baseScorer = new VectorScorer() {
            @Override
            public float score() throws IOException {
                int doc = vectorIter.docID();
                Float s = sparseScores.get(doc);
                if (s == null) {
                    throw new IllegalStateException("No score for doc " + doc);
                }
                return s;
            }

            @Override
            public DocIdSetIterator iterator() {
                return vectorIter;
            }
        };

        // Filter accepts 0, 6, 7, 8, 10 — but only 0 and 10 have vectors
        FixedBitSet filterBits = new FixedBitSet(NUM_DOCS);
        for (int i : new int[] { 0, 6, 7, 8, 10 }) {
            filterBits.set(i);
        }
        DocIdSetIterator filterIter = new BitSetIterator(filterBits, filterBits.length());

        NestedBestChildVectorScorer scorer = new NestedBestChildVectorScorer(filterIter, PARENT_BIT_SET, baseScorer);
        DocIdSetIterator iterator = scorer.iterator();

        // Parent 5: only child 0 (0.4)
        assertEquals(0, iterator.nextDoc());
        assertEquals(0.4f, scorer.score(), 0.001f);

        // Parent 9: skipped entirely — children 6,7,8 accepted by filter but none have vectors
        // Parent 11: child 10 (0.7)
        assertEquals(10, iterator.nextDoc());
        assertEquals(0.7f, scorer.score(), 0.001f);

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    // ──────────────────────────────────────────────
    // Helpers
    // ──────────────────────────────────────────────

    /** All child doc ids that have vectors: {0,1,2,3,4,6,7,8,10}. */
    private static FixedBitSet vectorBitSet() {
        FixedBitSet bits = new FixedBitSet(NUM_DOCS);
        SCORES.keySet().forEach(bits::set);
        return bits;
    }

    /** Accepted children filter: all vector children except 2 and 7. */
    private static FixedBitSet filterBitSet() {
        FixedBitSet bits = vectorBitSet();
        bits.clear(2);
        bits.clear(7);
        return bits;
    }

    /**
     * Creates a {@link NestedBestChildVectorScorer} using real {@link BitSetIterator}s.
     *
     * @param filterBits if non-null, used as the accepted children filter
     */
    private static NestedBestChildVectorScorer createScorer(FixedBitSet filterBits) {
        FixedBitSet vectorBits = vectorBitSet();
        DocIdSetIterator vectorIter = new BitSetIterator(vectorBits, vectorBits.length());

        VectorScorer baseScorer = new VectorScorer() {
            @Override
            public float score() throws IOException {
                int doc = vectorIter.docID();
                Float s = SCORES.get(doc);
                if (s == null) {
                    throw new IllegalStateException("No score for doc " + doc);
                }
                return s;
            }

            @Override
            public DocIdSetIterator iterator() {
                return vectorIter;
            }
        };

        DocIdSetIterator filterIter = filterBits != null ? new BitSetIterator(filterBits, filterBits.length()) : null;
        return new NestedBestChildVectorScorer(filterIter, PARENT_BIT_SET, baseScorer);
    }
}

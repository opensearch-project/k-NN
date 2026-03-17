/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.scorers;

import org.apache.lucene.search.ConjunctionUtils;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.util.BitSet;
import org.opensearch.common.Nullable;

import java.util.Arrays;

import java.io.IOException;

/**
 * A {@link VectorScorer} decorator for nested (parent-child) document structures that groups
 * child documents by their parent and yields only the highest-scoring child per parent.
 *
 * <p>This is adapted from Lucene's {@code DiversifyingChildrenVectorScorer} inner class in
 * {@code DiversifyingChildrenFloatKnnVectorQuery}, re-implemented as a standalone {@link VectorScorer}
 * so it can be used in OpenSearch's exact search path.
 *
 * <h2>Document Layout</h2>
 * <p>Lucene block-joins store parent and child documents in contiguous doc-id ranges:
 * <pre>
 *   [child_0, child_1, ..., child_n, PARENT, child_0, child_1, ..., child_m, PARENT, ...]
 * </pre>
 * The {@code parentBitSet} identifies which doc ids are parents. Every doc id between two
 * consecutive parent bits is a child of the later parent.
 *
 * <h2>Iteration Behavior</h2>
 * <p>Each call to {@link #iterator()}'s {@code nextDoc()} advances through one parent group:
 * <ol>
 *   <li>Finds the next child document (respecting the optional filter).</li>
 *   <li>Determines the parent for that child via {@code parentBitSet.nextSetBit()}.</li>
 *   <li>Iterates over all children belonging to that parent, scoring each one.</li>
 *   <li>Returns the doc id of the best-scoring child; {@link #score()} returns its score.</li>
 * </ol>
 *
 * <h2>Filtered vs Unfiltered</h2>
 * <ul>
 *   <li><b>Unfiltered</b> ({@code acceptedChildrenIterator == null}): every vector document is
 *       considered. The underlying vector iterator drives iteration directly.</li>
 *   <li><b>Filtered</b>: the {@code filterIdsIterator} is intersected with the vector
 *       iterator via {@link #maybeIntersectWithFilter}, producing a single iterator
 *       that yields only doc ids present in both. This keeps the two iterators in lockstep
 *       so the vector scorer is always positioned correctly when {@link #score()} is called.</li>
 * </ul>
 *
 * <h2>Example</h2>
 * <p>Given children [0,1,2,3,4] → parent 5, children [6,7,8] → parent 9, child [10] → parent 11,
 * and a filter that excludes children 2 and 7:
 * <pre>
 *   Accepted children: {0, 1, 3, 4, 6, 8, 10}
 *
 *   nextDoc() → bestChild=1  (best of {0,1,3,4} under parent 5)
 *   nextDoc() → bestChild=6  (best of {6,8} under parent 9)
 *   nextDoc() → bestChild=10 (only child under parent 11)
 *   nextDoc() → NO_MORE_DOCS
 * </pre>
 *
 * @see org.apache.lucene.search.VectorScorer
 * @see org.apache.lucene.search.join.DiversifyingChildrenFloatKnnVectorQuery
 */
class NestedBestChildVectorScorer implements VectorScorer {
    private final VectorScorer childrenVectorScorer;
    private final DocIdSetIterator childIterator;
    private final BitSet parentBitSet;
    private final DocIdSetIterator iterator;
    private int bestChild = -1;
    private float currentScore = Float.NEGATIVE_INFINITY;

    /**
     * Creates a scorer that finds the best-scoring child per parent, optionally restricted to a
     * subset of accepted children.
     *
     * <p>When {@code filterIdsIterator} is {@code null} (unfiltered), the scorer's own
     * vector iterator is used to drive child iteration, matching the behavior of Lucene's
     * {@code DiversifyingChildrenVectorScorer} but without requiring a separate filter iterator.
     *
     * @param filterIdsIterator iterator over the accepted child doc ids (i.e. children that
     *                                 pass the filter). Pass {@code null} for the unfiltered case
     *                                 where all vector documents are considered.
     * @param parentBitSet             a {@link BitSet} with bits set at every parent doc id.
     *                                 Used to determine parent boundaries for grouping children.
     * @param childrenVectorScorer             the underlying scorer that computes similarity scores for
     *                                 individual child documents against the query vector.
     */
    public NestedBestChildVectorScorer(
        @Nullable DocIdSetIterator filterIdsIterator,
        BitSet parentBitSet,
        VectorScorer childrenVectorScorer
    ) {
        this.childrenVectorScorer = childrenVectorScorer;
        this.parentBitSet = parentBitSet;
        DocIdSetIterator vectorIterator = childrenVectorScorer.iterator();
        this.childIterator = maybeIntersectWithFilter(vectorIterator, filterIdsIterator);
        this.iterator = createIterator();
    }

    /**
     * Returns the score of the best-scoring child for the current parent group.
     * Only valid after a successful call to {@code iterator().nextDoc()}.
     */
    @Override
    public float score() throws IOException {
        return currentScore;
    }

    /**
     * Returns a {@link DocIdSetIterator} whose {@code nextDoc()} yields the doc id of the
     * best-scoring child for each successive parent. The same instance is returned on every call.
     */
    @Override
    public DocIdSetIterator iterator() {
        return iterator;
    }

    /**
     * Returns the vector iterator directly if no filter is provided, otherwise intersects
     * it with the filter so that only doc ids present in both are yielded.
     */
    private static DocIdSetIterator maybeIntersectWithFilter(
        DocIdSetIterator vectorIterator,
        @Nullable DocIdSetIterator filterIdsIterator
    ) {
        if (filterIdsIterator == null) {
            return vectorIterator;
        }
        return ConjunctionUtils.intersectIterators(Arrays.asList(filterIdsIterator, vectorIterator));
    }

    /**
     * Creates a {@link DocIdSetIterator} that groups children by parent and yields the
     * best-scoring child per parent. Each {@code nextDoc()} call advances through one
     * parent group and returns the doc id of the highest-scoring child within that group.
     */
    private DocIdSetIterator createIterator() {
        return new DocIdSetIterator() {
            @Override
            public int docID() {
                return bestChild;
            }

            @Override
            public int nextDoc() throws IOException {
                int nextChild = childIterator.docID();
                if (nextChild == -1) {
                    nextChild = childIterator.nextDoc();
                }
                if (nextChild == NO_MORE_DOCS) {
                    bestChild = NO_MORE_DOCS;
                    return NO_MORE_DOCS;
                }

                currentScore = Float.NEGATIVE_INFINITY;
                int currentParent = parentBitSet.nextSetBit(nextChild);

                do {
                    float score = childrenVectorScorer.score();
                    if (score > currentScore) {
                        bestChild = nextChild;
                        currentScore = score;
                    }
                } while ((nextChild = childIterator.nextDoc()) != NO_MORE_DOCS && nextChild < currentParent);

                return bestChild;
            }

            /**
             * Not supported. This iterator returns the best-scoring child per parent group,
             * which requires evaluating <em>all</em> children within a group. Advancing to an
             * arbitrary target could land in the middle of a parent group, making it impossible
             * to consider earlier (potentially higher-scoring) children without backtracking
             * — violating the forward-only iterator contract.
             */
            @Override
            public int advance(int target) {
                throw new UnsupportedOperationException();
            }

            @Override
            public long cost() {
                return childIterator.cost();
            }
        };
    }
}

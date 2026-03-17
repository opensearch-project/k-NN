/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.exactsearch.scorers;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.util.BitSet;
import org.opensearch.common.Nullable;

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
 *       considered. The underlying {@code vectorIterator} drives iteration directly.</li>
 *   <li><b>Filtered</b>: only children present in the {@code acceptedChildrenIterator} are
 *       visited. The {@code vectorIterator} is advanced to each accepted child to compute its
 *       score, ensuring that {@code vectorIterator.advance()} is never called for filtered-out
 *       children.</li>
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
public class NestedBestChildVectorScorer implements VectorScorer {
    private final VectorScorer vectorScorer;
    private final DocIdSetIterator vectorIterator;
    private final DocIdSetIterator childIterator;
    private final BitSet parentBitSet;
    private final DocIdSetIterator iterator;
    private int bestChild = -1;
    private float currentScore = Float.NEGATIVE_INFINITY;

    /**
     * Creates a scorer that finds the best-scoring child per parent, optionally restricted to a
     * subset of accepted children.
     *
     * @param acceptedChildrenIterator iterator over the accepted child doc ids (i.e. children that
     *                                 pass the filter). Pass {@code null} for the unfiltered case
     *                                 where all vector documents are considered.
     * @param parentBitSet             a {@link BitSet} with bits set at every parent doc id.
     *                                 Used to determine parent boundaries for grouping children.
     * @param vectorScorer             the underlying scorer that computes similarity scores for
     *                                 individual child documents against the query vector.
     */
    public NestedBestChildVectorScorer(
        @Nullable DocIdSetIterator acceptedChildrenIterator,
        BitSet parentBitSet,
        VectorScorer vectorScorer
    ) {
        this.vectorScorer = vectorScorer;
        this.vectorIterator = vectorScorer.iterator();
        this.childIterator = acceptedChildrenIterator;
        this.parentBitSet = parentBitSet;
        this.iterator = createIterator();
    }

    /**
     * Convenience constructor for the unfiltered case where all child documents are considered.
     *
     * @param parentBitSet  a {@link BitSet} with bits set at every parent doc id.
     * @param vectorScorer  the underlying scorer for computing similarity scores.
     */
    public NestedBestChildVectorScorer(BitSet parentBitSet, VectorScorer vectorScorer) {
        this(null, parentBitSet, vectorScorer);
    }

    /**
     * Returns the score of the best-scoring child for the current parent group.
     * Only valid after a successful call to {@code iterator().nextDoc()}.
     *
     * @return the highest similarity score among the children of the current parent.
     */
    @Override
    public float score() throws IOException {
        return currentScore;
    }

    /**
     * Returns a {@link DocIdSetIterator} whose {@code nextDoc()} yields the doc id of the
     * best-scoring child for each successive parent. The same instance is returned on every call.
     *
     * @return the iterator over best-child doc ids.
     */
    @Override
    public DocIdSetIterator iterator() {
        return iterator;
    }

    private DocIdSetIterator createIterator() {
        return new DocIdSetIterator() {
            @Override
            public int docID() {
                return bestChild;
            }

            @Override
            public int nextDoc() throws IOException {
                int nextChild = nextChildDoc();
                if (nextChild == NO_MORE_DOCS) {
                    bestChild = NO_MORE_DOCS;
                    return NO_MORE_DOCS;
                }

                currentScore = Float.NEGATIVE_INFINITY;
                int currentParent = parentBitSet.nextSetBit(nextChild);
                if (currentParent == -1) {
                    bestChild = NO_MORE_DOCS;
                    return NO_MORE_DOCS;
                }

                do {
                    advanceVectorIterator(nextChild);
                    float score = vectorScorer.score();
                    if (score > currentScore) {
                        bestChild = nextChild;
                        currentScore = score;
                    }
                } while ((nextChild = nextChildDoc()) != NO_MORE_DOCS && nextChild < currentParent);

                return bestChild;
            }

            private int nextChildDoc() throws IOException {
                if (childIterator == null) {
                    int docId = vectorIterator.docID();
                    return docId == -1 ? vectorIterator.nextDoc() : docId;
                }
                int docId = childIterator.docID();
                return docId == -1 ? childIterator.nextDoc() : docId;
            }

            private void advanceVectorIterator(int target) throws IOException {
                if (childIterator == null) {
                    // Unfiltered: vectorIterator is already positioned via nextChildDoc
                    vectorIterator.nextDoc();
                } else {
                    vectorIterator.advance(target);
                    childIterator.nextDoc();
                }
            }

            @Override
            public int advance(int target) {
                throw new UnsupportedOperationException();
            }

            @Override
            public long cost() {
                return childIterator != null ? childIterator.cost() : vectorIterator.cost();
            }
        };
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.Weight;

import java.io.IOException;
import java.util.Map;

/**
 * <p>
 * <code>KNNScorer</code> exposes an {@link #iterator()} over documents
 * matching a query in increasing order of doc Id.
 * </p>
 * <p>
 * Document scores are computed using nmslib via JNI implementation.
 * </p>
 */
public class KNNScorer extends Scorer {

    private final DocIdSetIterator docIdsIter;
    private final Map<Integer, Float> scores;
    private final float boost;

    public KNNScorer(Weight weight, DocIdSetIterator docIdsIter, Map<Integer, Float> scores, float boost) {
        super();
        this.docIdsIter = docIdsIter;
        this.scores = scores;
        this.boost = boost;
    }

    @Override
    public DocIdSetIterator iterator() {
        return docIdsIter;
    }

    @Override
    public float getMaxScore(int upTo) throws IOException {
        return Float.MAX_VALUE;
    }

    @Override
    public float score() {
        assert docID() != DocIdSetIterator.NO_MORE_DOCS;
        Float score = scores.get(docID());
        if (score == null) throw new RuntimeException("Null score for the docID: " + docID());
        return score * boost;
    }

    @Override
    public int docID() {
        return docIdsIter.docID();
    }

    /**
     * Returns an Empty Scorer. We use this scorer to short circuit the actual search when it is not
     * required. Since the underlying DocIdSetIterator.empty() is stateful and not thread-safe we must create a new
     * scorer instance each time to avoid race conditions.
     * @return {@link KNNScorer}
     */
    public static Scorer emptyScorer() {
        return new Scorer() {
            private final DocIdSetIterator docIdsIter = DocIdSetIterator.empty();

            @Override
            public DocIdSetIterator iterator() {
                return docIdsIter;
            }

            @Override
            public float getMaxScore(int upTo) throws IOException {
                return 0;
            }

            @Override
            public float score() throws IOException {
                assert docID() != DocIdSetIterator.NO_MORE_DOCS;
                return 0;
            }

            @Override
            public int docID() {
                return docIdsIter.docID();
            }

        };
    }

    // private static final Scorer EMPTY_SCORER_INSTANCE = new Scorer() {
    // /**
    // * stateless empty DocIdSetIterator. Used in testing as opposed to DocIdSetIterator.empty() since
    // * DocIdSetIterator.empty() contains a stateful exhausted variable. If we associate a particular
    // * DocIdSetIterator.empty() instance with our static EMPTY_SCORER_INSTANCE then we hit an assertion error
    // * when multiple threads race to call advance() as it is not thread-safe.
    // */
    // public static DocIdSetIterator statelessEmptyDocIdSetIterator() {
    // return new DocIdSetIterator() {
    // public int advance(int target) {
    // return Integer.MAX_VALUE;
    // }
    //
    // public int docID() {
    // return DocIdSetIterator.NO_MORE_DOCS;
    // }
    //
    // public int nextDoc() {
    // return Integer.MAX_VALUE;
    // }
    //
    // public long cost() {
    // return 0L;
    // }
    // };
    // }
    //
    // private static final DocIdSetIterator docIdsIter = statelessEmptyDocIdSetIterator();
    //
    // @Override
    // public DocIdSetIterator iterator() {
    // return docIdsIter;
    // }
    //
    // @Override
    // public float getMaxScore(int upTo) throws IOException {
    // return 0;
    // }
    //
    // @Override
    // public float score() throws IOException {
    // assert docID() != DocIdSetIterator.NO_MORE_DOCS;
    // return 0;
    // }
    //
    // @Override
    // public int docID() {
    // return docIdsIter.docID();
    // }
    //
    // @Override
    // public boolean equals(Object obj) {
    // return this == obj; // Singleton ensures only one instance exists
    // }
    //
    // @Override
    // public int hashCode() {
    // return System.identityHashCode(this); // Consistent hash for singleton
    // }
    // };
}

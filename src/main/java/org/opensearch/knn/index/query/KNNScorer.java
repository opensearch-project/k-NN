/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Scorer;

import java.util.Collections;
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

    public KNNScorer(DocIdSetIterator docIdsIter, Map<Integer, Float> scores, float boost) {
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
    public float getMaxScore(int upTo) {
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
    public static KNNScorer emptyScorer() {
        return new KNNScorer(DocIdSetIterator.empty(), Collections.emptyMap(), 0);
    }
}

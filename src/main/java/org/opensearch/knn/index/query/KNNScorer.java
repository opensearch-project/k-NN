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
        super(weight);
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
        return score;
    }

    @Override
    public int docID() {
        return docIdsIter.docID();
    }
}

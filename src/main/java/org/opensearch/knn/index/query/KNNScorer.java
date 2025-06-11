/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;

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

    private final float boost;
    private final TopDocsDISI docIdsIter;

    public KNNScorer(TopDocs topDocs, final float boost) {
        super();
        this.boost = boost;
        this.docIdsIter = new TopDocsDISI(topDocs);
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
        return docIdsIter.score() * boost;
    }

    @Override
    public int docID() {
        return docIdsIter.docID();
    }

    /**
     * Returns the Empty Scorer implementation. We use this scorer to short circuit the actual search when it is not
     * required.
     * @return {@link KNNScorer}
     */
    public static Scorer emptyScorer() {
        return new KNNScorer(TopDocsCollector.EMPTY_TOPDOCS, 0);
    }
}

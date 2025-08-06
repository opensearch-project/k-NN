/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Scorer;
import org.opensearch.knn.index.query.iterators.KNNIterator;

import java.io.IOException;

/**
 * Lazy scorer for exact k-NN search that computes vector similarities on-demand.
 * Unlike KNNScorer which uses pre-computed TopDocs, this scorer calculates scores
 * only when requested, preventing OOM exceptions for large document sets.
 */
@Log4j2
public class KNNLazyScorer extends Scorer {
    private final float boost;
    private final KNNIterator knnIterator;
    private int currentDoc = -1;

    public KNNLazyScorer(KNNIterator knnIterator, float boost) throws IOException {
        super();
        this.knnIterator = knnIterator;
        this.boost = boost;
    }

    @Override
    public DocIdSetIterator iterator() {
        return new DocIdSetIterator() {
            @Override
            public int docID() {
                return currentDoc;
            }

            @Override
            public int nextDoc() throws IOException {
                currentDoc = knnIterator.nextDoc();
                if (currentDoc < 0 && currentDoc != NO_MORE_DOCS) {
                    currentDoc = NO_MORE_DOCS;
                }
                return currentDoc;
            }

            @Override
            public int advance(int target) throws IOException {
                while (currentDoc < target) {
                    currentDoc = knnIterator.nextDoc();
                    if (currentDoc == NO_MORE_DOCS) {
                        break;
                    }
                }
                return currentDoc;
            }

            @Override
            public long cost() {
                return 0;
            }
        };
    }

    @Override
    public float score() throws IOException {
        return knnIterator.score() * boost;
    }

    @Override
    public int docID() {
        return currentDoc;
    }

    @Override
    public float getMaxScore(int upTo) throws IOException {
        return Float.MAX_VALUE;
    }
}

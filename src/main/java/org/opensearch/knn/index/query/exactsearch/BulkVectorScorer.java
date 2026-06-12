/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.exactsearch;

import org.apache.lucene.search.DocAndFloatFeatureBuffer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.VectorScorer;

import java.io.IOException;
import java.util.function.Predicate;

/**
 * A {@link Scorer} that scores documents using bulk vector scoring, yielding only those
 * whose score satisfies the provided {@link Predicate}.
 */
public class BulkVectorScorer extends Scorer {

    private final DocAndFloatFeatureBuffer buffer = new DocAndFloatFeatureBuffer();
    private final VectorScorer.Bulk bulkScorer;
    private final Predicate<Float> scoreFilter;
    private final long cost;

    private int currentDocId = -1;
    private int currentBatchIdx = 0;
    private float currentScore;

    public BulkVectorScorer(final VectorScorer vectorScorer, final DocIdSetIterator matchedDocs, final Predicate<Float> scoreFilter)
        throws IOException {
        this.bulkScorer = vectorScorer.bulk(matchedDocs);
        this.scoreFilter = scoreFilter;
        this.cost = matchedDocs != null ? matchedDocs.cost() : vectorScorer.iterator().cost();
    }

    public static BulkVectorScorer fullPrecision(VectorScorer vectorScorer, DocIdSetIterator matchedDocs) throws IOException {
        return new BulkVectorScorer(vectorScorer, matchedDocs, score -> true);
    }

    public static BulkVectorScorer fullPrecision(VectorScorer vectorScorer, DocIdSetIterator matchedDocs, float minScore)
        throws IOException {
        return new BulkVectorScorer(vectorScorer, matchedDocs, score -> score >= minScore);
    }

    @Override
    public int docID() {
        return currentDocId;
    }

    @Override
    public DocIdSetIterator iterator() {
        return new DocIdSetIterator() {

            @Override
            public int docID() {
                return currentDocId;
            }

            @Override
            public int nextDoc() throws IOException {
                while (true) {
                    int result = scanBufferForMatch();
                    if (result != -1) {
                        return result;
                    }
                    float maxBatchScore = bulkScorer.nextDocsAndScores(DocIdSetIterator.NO_MORE_DOCS, null, buffer);
                    currentBatchIdx = 0;
                    if (buffer.size == 0) {
                        return currentDocId = NO_MORE_DOCS;
                    }
                    if (!scoreFilter.test(maxBatchScore)) {
                        currentBatchIdx = buffer.size;
                    }
                }
            }

            @Override
            public int advance(int target) throws IOException {
                while (true) {
                    int doc = nextDoc();
                    if (doc == NO_MORE_DOCS) {
                        return NO_MORE_DOCS;
                    }
                    if (doc >= target) {
                        return doc;
                    }
                }
            }

            @Override
            public long cost() {
                return cost;
            }
        };
    }

    @Override
    public float getMaxScore(int upTo) {
        return Float.MAX_VALUE;
    }

    @Override
    public float score() {
        return currentScore;
    }

    private int scanBufferForMatch() {
        while (currentBatchIdx < buffer.size) {
            if (scoreFilter.test(buffer.features[currentBatchIdx])) {
                currentDocId = buffer.docs[currentBatchIdx];
                currentScore = buffer.features[currentBatchIdx];
                currentBatchIdx++;
                return currentDocId;
            }
            currentBatchIdx++;
        }
        return -1;
    }
}

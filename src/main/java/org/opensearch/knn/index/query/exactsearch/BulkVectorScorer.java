/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.exactsearch;

import org.apache.lucene.index.QueryTimeout;
import org.apache.lucene.search.DocAndFloatFeatureBuffer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.VectorScorer;
import org.opensearch.common.Nullable;

import java.io.IOException;
import java.util.function.Predicate;

/**
 * A {@link Scorer} that scores documents using bulk vector scoring, yielding only those
 * whose score satisfies the provided {@link Predicate}. When given a {@link QueryTimeout}, it checks the timeout once
 * per scored batch and stops iterating once it fires, so a cancelled or timed-out search keeps the results collected
 * so far.
 */
public class BulkVectorScorer extends Scorer {

    private final DocAndFloatFeatureBuffer buffer = new DocAndFloatFeatureBuffer();
    private final VectorScorer.Bulk bulkScorer;
    private final Predicate<Float> scoreFilter;
    @Nullable
    private final QueryTimeout queryTimeout;
    private final long cost;

    private int currentDocId = -1;
    private int currentBatchIdx = 0;
    private float currentScore;
    private float minCompetitiveScore = 0f;

    private BulkVectorScorer(
        final VectorScorer vectorScorer,
        final DocIdSetIterator matchedDocs,
        final Predicate<Float> scoreFilter,
        @Nullable final QueryTimeout queryTimeout
    ) throws IOException {
        this.bulkScorer = vectorScorer.bulk(matchedDocs);
        this.scoreFilter = scoreFilter;
        this.queryTimeout = queryTimeout;
        this.cost = matchedDocs != null ? matchedDocs.cost() : vectorScorer.iterator().cost();
    }

    public static BulkVectorScorer forKSearch(VectorScorer vectorScorer, DocIdSetIterator matchedDocs) throws IOException {
        return forKSearch(vectorScorer, matchedDocs, null);
    }

    public static BulkVectorScorer forKSearch(VectorScorer vectorScorer, DocIdSetIterator matchedDocs, @Nullable QueryTimeout queryTimeout)
        throws IOException {
        return new BulkVectorScorer(vectorScorer, matchedDocs, score -> true, queryTimeout);
    }

    public static BulkVectorScorer forRadialSearch(VectorScorer vectorScorer, DocIdSetIterator matchedDocs, float minScore)
        throws IOException {
        return forRadialSearch(vectorScorer, matchedDocs, minScore, null);
    }

    public static BulkVectorScorer forRadialSearch(
        VectorScorer vectorScorer,
        DocIdSetIterator matchedDocs,
        float minScore,
        @Nullable QueryTimeout queryTimeout
    ) throws IOException {
        return new BulkVectorScorer(vectorScorer, matchedDocs, score -> score >= minScore, queryTimeout);
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
                    if (queryTimeout != null && queryTimeout.shouldExit()) {
                        return currentDocId = NO_MORE_DOCS;
                    }
                    float maxBatchScore = bulkScorer.nextDocsAndScores(DocIdSetIterator.NO_MORE_DOCS, null, buffer);
                    currentBatchIdx = 0;
                    if (buffer.size == 0) {
                        return currentDocId = NO_MORE_DOCS;
                    }
                    if (!scoreFilter.test(maxBatchScore) || maxBatchScore < minCompetitiveScore) {
                        currentBatchIdx = buffer.size;
                    }
                }
            }

            @Override
            public int advance(int target) throws IOException {
                if (currentDocId >= target) {
                    return currentDocId;
                }
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
    public void setMinCompetitiveScore(float score) {
        this.minCompetitiveScore = score;
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
            float score = buffer.features[currentBatchIdx];
            if (scoreFilter.test(score) && score >= minCompetitiveScore) {
                currentDocId = buffer.docs[currentBatchIdx];
                currentScore = score;
                currentBatchIdx++;
                return currentDocId;
            }
            currentBatchIdx++;
        }
        return -1;
    }
}

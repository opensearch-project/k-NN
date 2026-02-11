/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.exactsearch;

import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.query.SegmentLevelQuantizationInfo;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;

import org.opensearch.knn.plugin.script.KNNScoringUtil;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;

import java.io.IOException;

/**
 * Inspired by DiversifyingChildrenFloatKnnVectorQuery in lucene
 * https://github.com/apache/lucene/blob/7b8aece125aabff2823626d5b939abf4747f63a7/lucene/join/src/java/org/apache/lucene/search/join/DiversifyingChildrenFloatKnnVectorQuery.java#L162
 *
 * The class is used in KNNWeight to score all docs, but, it iterates over filterIdsArray if filter is provided
 */
class VectorIdsExactKNNIterator implements ExactKNNIterator {
    protected final DocIdSetIterator filterIdsIterator;
    protected final float[] queryVector;
    protected final KNNFloatVectorValues knnFloatVectorValues;
    protected final SpaceType spaceType;
    protected float currentScore = Float.NEGATIVE_INFINITY;
    protected int docId;

    public VectorIdsExactKNNIterator(
        @Nullable final DocIdSetIterator filterIdsIterator,
        final float[] queryVector,
        final KNNFloatVectorValues knnFloatVectorValues,
        final SpaceType spaceType
    ) throws IOException {
        this(filterIdsIterator, queryVector, knnFloatVectorValues, spaceType, null, null);
    }

    public VectorIdsExactKNNIterator(final float[] queryVector, final KNNFloatVectorValues knnFloatVectorValues, final SpaceType spaceType)
        throws IOException {
        this(null, queryVector, knnFloatVectorValues, spaceType);
    }

    public VectorIdsExactKNNIterator(
        @Nullable final DocIdSetIterator filterIdsIterator,
        final float[] queryVector,
        final KNNFloatVectorValues knnFloatVectorValues,
        final SpaceType spaceType,
        final byte[] quantizedQueryVector,
        final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo
    ) throws IOException {
        this.filterIdsIterator = filterIdsIterator;
        this.queryVector = queryVector;
        this.knnFloatVectorValues = knnFloatVectorValues;
        this.spaceType = spaceType;
        // This cannot be moved inside nextDoc() method since it will break when we have nested field, where
        // nextDoc should already be referring to next knnVectorValues
        this.docId = getNextDocId();
    }

    /**
     * Advance to the next doc and update score value with score of the next doc.
     * DocIdSetIterator.NO_MORE_DOCS is returned when there is no more docs
     *
     * @return next doc id
     */
    @Override
    public int nextDoc() throws IOException {

        if (docId == DocIdSetIterator.NO_MORE_DOCS) {
            return DocIdSetIterator.NO_MORE_DOCS;
        }
        currentScore = computeScore();
        int currentDocId = docId;
        docId = getNextDocId();
        return currentDocId;
    }

    @Override
    public float score() {
        return currentScore;
    }

    protected float computeScore() throws IOException {
        final float[] vector = knnFloatVectorValues.getVector();
        return spaceType.getKnnVectorSimilarityFunction().compare(queryVector, vector);
    }

    protected int getNextDocId() throws IOException {
        if (filterIdsIterator == null) {
            return knnFloatVectorValues.nextDoc();
        }
        int nextDocID = this.filterIdsIterator.nextDoc();
        // For filter case, advance vector values to corresponding doc id from filter bit set
        if (nextDocID != DocIdSetIterator.NO_MORE_DOCS) {
            knnFloatVectorValues.advance(nextDocID);
        }
        return nextDocID;
    }

    /*
        protected for testing.
        Logic:
        - segmentLevelQuantizationInfo is null -> should not score with ADC
        - quantizationParams is not ScalarQuantizationParams -> should not score with ADC
        - quantizationParams is ScalarQuantizationParams -> defer to isEnableADC() to determine if should score with ADC.
     */
    protected boolean shouldScoreWithADC(SegmentLevelQuantizationInfo segmentLevelQuantizationInfo) {
        if (segmentLevelQuantizationInfo == null) {
            return false;
        }

        if (segmentLevelQuantizationInfo.getQuantizationParams() instanceof ScalarQuantizationParams scalarQuantizationParams) {
            return scalarQuantizationParams.isEnableADC();
        }
        return false;
    }

    // protected for testing. scoreWithADC is used in exact searcher.
    protected float scoreWithADC(float[] queryVector, byte[] documentVector, SpaceType spaceType) {
        // NOTE: the prescore translations come from Faiss.java::SCORE_TRANSLATIONS.
        if (spaceType.equals(SpaceType.L2)) {
            return SpaceType.L2.scoreTranslation(KNNScoringUtil.l2SquaredADC(queryVector, documentVector));
        } else if (spaceType.equals(SpaceType.INNER_PRODUCT)) {
            return SpaceType.INNER_PRODUCT.scoreTranslation((-1 * KNNScoringUtil.innerProductADC(queryVector, documentVector)));
        } else if (spaceType.equals(SpaceType.COSINESIMIL)) {
            return SpaceType.COSINESIMIL.scoreTranslation(1 - KNNScoringUtil.innerProductADC(queryVector, documentVector));
        }

        throw new UnsupportedOperationException("Space type " + spaceType.getValue() + " is not supported for ADC");
    }
}

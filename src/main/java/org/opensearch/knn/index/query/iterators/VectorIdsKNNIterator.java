/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.iterators;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.query.SegmentLevelQuantizationInfo;
import org.opensearch.knn.index.query.SegmentLevelQuantizationUtil;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;

import java.io.IOException;

/**
 * Inspired by DiversifyingChildrenFloatKnnVectorQuery in lucene
 * https://github.com/apache/lucene/blob/7b8aece125aabff2823626d5b939abf4747f63a7/lucene/join/src/java/org/apache/lucene/search/join/DiversifyingChildrenFloatKnnVectorQuery.java#L162
 *
 * The class is used in KNNWeight to score all docs, but, it iterates over filterIdsArray if filter is provided
 */
public class VectorIdsKNNIterator implements KNNIterator {
    // Array of doc ids to iterate
    protected final BitSet filterIdsBitSet;
    protected final BitSetIterator bitSetIterator;
    protected final float[] queryVector;
    private final byte[] quantizedQueryVector;
    protected final KNNFloatVectorValues knnFloatVectorValues;
    protected final SpaceType spaceType;
    protected float currentScore = Float.NEGATIVE_INFINITY;
    protected int docId;
    private final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo;

    VectorIdsKNNIterator(
        final BitSet filterIdsBitSet,
        final float[] queryVector,
        final KNNFloatVectorValues knnFloatVectorValues,
        final SpaceType spaceType
    ) {
        this(filterIdsBitSet, queryVector, knnFloatVectorValues, spaceType, null, null);
    }

    public VectorIdsKNNIterator(
        final BitSet filterIdsBitSet,
        final float[] queryVector,
        final KNNFloatVectorValues knnFloatVectorValues,
        final SpaceType spaceType,
        final byte[] quantizedQueryVector,
        final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo
    ) {
        this.filterIdsBitSet = filterIdsBitSet;
        this.bitSetIterator = new BitSetIterator(filterIdsBitSet, filterIdsBitSet.length());
        this.queryVector = queryVector;
        this.knnFloatVectorValues = knnFloatVectorValues;
        this.spaceType = spaceType;
        this.docId = bitSetIterator.nextDoc();
        this.quantizedQueryVector = quantizedQueryVector;
        this.segmentLevelQuantizationInfo = segmentLevelQuantizationInfo;
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
        int doc = knnFloatVectorValues.advance(docId);
        currentScore = computeScore();
        docId = bitSetIterator.nextDoc();
        return doc;
    }

    @Override
    public float score() {
        return currentScore;
    }

    protected float computeScore() throws IOException {
        final float[] vector = knnFloatVectorValues.getVector();
        if (segmentLevelQuantizationInfo != null && quantizedQueryVector != null) {
            byte[] quantizedVector = SegmentLevelQuantizationUtil.quantizeVector(vector, segmentLevelQuantizationInfo);
            return SpaceType.HAMMING.getKnnVectorSimilarityFunction().compare(quantizedQueryVector, quantizedVector);
        } else {
            // Calculates a similarity score between the two vectors with a specified function. Higher similarity
            // scores correspond to closer vectors.
            return spaceType.getKnnVectorSimilarityFunction().compare(queryVector, vector);
        }
    }
}

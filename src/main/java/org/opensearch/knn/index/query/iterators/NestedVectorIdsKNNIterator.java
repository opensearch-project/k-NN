/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.iterators;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BitSet;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.query.SegmentLevelQuantizationInfo;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;

import java.io.IOException;

/**
 * This iterator iterates filterIdsArray to score if filter is provided else it iterates over all docs.
 * However, it dedupe docs per each parent doc
 * of which ID is set in parentBitSet and only return best child doc with the highest score.
 */
public class NestedVectorIdsKNNIterator extends VectorIdsKNNIterator {
    private final BitSet parentBitSet;

    public NestedVectorIdsKNNIterator(
        @Nullable final BitSet filterIdsArray,
        final float[] queryVector,
        final KNNFloatVectorValues knnFloatVectorValues,
        final SpaceType spaceType,
        final BitSet parentBitSet
    ) throws IOException {
        this(filterIdsArray, queryVector, knnFloatVectorValues, spaceType, parentBitSet, null, null);
    }

    public NestedVectorIdsKNNIterator(
        final float[] queryVector,
        final KNNFloatVectorValues knnFloatVectorValues,
        final SpaceType spaceType,
        final BitSet parentBitSet
    ) throws IOException {
        this(null, queryVector, knnFloatVectorValues, spaceType, parentBitSet, null, null);
    }

    public NestedVectorIdsKNNIterator(
        @Nullable final BitSet filterIdsArray,
        final float[] queryVector,
        final KNNFloatVectorValues knnFloatVectorValues,
        final SpaceType spaceType,
        final BitSet parentBitSet,
        final byte[] quantizedVector,
        final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo
    ) throws IOException {
        super(filterIdsArray, queryVector, knnFloatVectorValues, spaceType, quantizedVector, segmentLevelQuantizationInfo);
        this.parentBitSet = parentBitSet;
    }

    /**
     * Advance to the next best child doc per parent and update score with the best score among child docs from the parent.
     * DocIdSetIterator.NO_MORE_DOCS is returned when there is no more docs
     *
     * @return next best child doc id
     */
    @Override
    public int nextDoc() throws IOException {
        if (docId == DocIdSetIterator.NO_MORE_DOCS) {
            return DocIdSetIterator.NO_MORE_DOCS;
        }

        currentScore = Float.NEGATIVE_INFINITY;
        int currentParent = parentBitSet.nextSetBit(docId);
        int bestChild = -1;

        // In order to traverse all children for given parent, we have to use docId < parentId, because,
        // kNNVectorValues will not have parent id since DocId is unique per segment. For ex: let's say for doc id 1, there is one child
        // and for doc id 5, there are three children. In that case knnVectorValues iterator will have [0, 2, 3, 4]
        // and parentBitSet will have [1,5]
        // Hence, we have to iterate till docId from knnVectorValues is less than parentId instead of till equal to parentId
        while (docId != DocIdSetIterator.NO_MORE_DOCS && docId < currentParent) {
            float score = computeScore();
            if (score > currentScore) {
                bestChild = docId;
                currentScore = score;
            }
            docId = getNextDocId();
        }

        return bestChild;
    }
}

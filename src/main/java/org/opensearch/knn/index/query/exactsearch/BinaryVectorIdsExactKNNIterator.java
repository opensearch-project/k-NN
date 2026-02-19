/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.exactsearch;

import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.vectorvalues.KNNBinaryVectorValues;
import org.opensearch.knn.plugin.script.KNNScoringUtil;

import java.io.IOException;

/**
 * Inspired by DiversifyingChildrenFloatKnnVectorQuery in lucene
 * https://github.com/apache/lucene/blob/7b8aece125aabff2823626d5b939abf4747f63a7/lucene/join/src/java/org/apache/lucene/search/join/DiversifyingChildrenFloatKnnVectorQuery.java#L162
 *
 * The class is used in KNNWeight to score all docs, but, it iterates over filterIdsArray if filter is provided
 */
class BinaryVectorIdsExactKNNIterator implements ExactKNNIterator {
    protected final DocIdSetIterator docIdSetIterator;
    protected final byte[] byteQueryVector;
    protected final float[] floatQueryVector;
    protected final KNNBinaryVectorValues binaryVectorValues;
    protected final SpaceType spaceType;
    protected float currentScore = Float.NEGATIVE_INFINITY;
    protected int docId;

    public BinaryVectorIdsExactKNNIterator(
        @Nullable final DocIdSetIterator docIdSetIterator,
        final byte[] byteQueryVector,
        final KNNBinaryVectorValues binaryVectorValues,
        final SpaceType spaceType
    ) throws IOException {
        this(docIdSetIterator, byteQueryVector, null, binaryVectorValues, spaceType);
    }

    public BinaryVectorIdsExactKNNIterator(
        @Nullable final DocIdSetIterator docIdSetIterator,
        final float[] floatQueryVector,
        final KNNBinaryVectorValues binaryVectorValues,
        final SpaceType spaceType
    ) throws IOException {
        this(docIdSetIterator, null, floatQueryVector, binaryVectorValues, spaceType);
    }

    private BinaryVectorIdsExactKNNIterator(
        @Nullable final DocIdSetIterator docIdSetIterator,
        final byte[] byteQueryVector,
        final float[] floatQueryVector,
        final KNNBinaryVectorValues binaryVectorValues,
        final SpaceType spaceType
    ) throws IOException {
        assert (floatQueryVector == null) != (byteQueryVector == null)
            : "Exactly one of byteQueryVector or floatQueryVector must be non-null";
        this.docIdSetIterator = docIdSetIterator;
        this.byteQueryVector = byteQueryVector;
        this.floatQueryVector = floatQueryVector;
        this.binaryVectorValues = binaryVectorValues;
        this.spaceType = spaceType;
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
        final byte[] documentVector = binaryVectorValues.getVector();
        if (floatQueryVector != null) {
            return KNNScoringUtil.scoreWithADC(floatQueryVector, documentVector, spaceType);
        }
        return spaceType.getKnnVectorSimilarityFunction().compare(byteQueryVector, documentVector);
    }

    protected int getNextDocId() throws IOException {
        if (docIdSetIterator == null) {
            return binaryVectorValues.nextDoc();
        }
        int nextDocID = this.docIdSetIterator.nextDoc();
        // For filter case, advance vector values to corresponding doc id from filter bit set
        if (nextDocID != DocIdSetIterator.NO_MORE_DOCS) {
            binaryVectorValues.advance(nextDocID);
        }
        return nextDocID;
    }
}

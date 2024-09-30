/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.iterators;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.vectorvalues.KNNByteVectorValues;

import java.io.IOException;

/**
 * Inspired by DiversifyingChildrenFloatKnnVectorQuery in lucene
 * https://github.com/apache/lucene/blob/7b8aece125aabff2823626d5b939abf4747f63a7/lucene/join/src/java/org/apache/lucene/search/join/DiversifyingChildrenFloatKnnVectorQuery.java#L162
 *
 * The class is used in KNNWeight to score all docs, but, it iterates over filterIdsArray if filter is provided
 */
public class ByteVectorIdsKNNIterator implements KNNIterator {
    protected final BitSetIterator bitSetIterator;
    protected final float[] queryVector;
    protected final KNNByteVectorValues byteVectorValues;
    protected final SpaceType spaceType;
    protected float currentScore = Float.NEGATIVE_INFINITY;
    protected int docId;

    public ByteVectorIdsKNNIterator(
        @Nullable final BitSet filterIdsBitSet,
        final float[] queryVector,
        final KNNByteVectorValues byteVectorValues,
        final SpaceType spaceType
    ) throws IOException {
        this.bitSetIterator = filterIdsBitSet == null ? null : new BitSetIterator(filterIdsBitSet, filterIdsBitSet.length());
        this.queryVector = queryVector;
        this.byteVectorValues = byteVectorValues;
        this.spaceType = spaceType;
        // This cannot be moved inside nextDoc() method since it will break when we have nested field, where
        // nextDoc should already be referring to next knnVectorValues
        this.docId = getNextDocId();
    }

    public ByteVectorIdsKNNIterator(final float[] queryVector, final KNNByteVectorValues byteVectorValues, final SpaceType spaceType)
        throws IOException {
        this(null, queryVector, byteVectorValues, spaceType);
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
        final byte[] vector = byteVectorValues.getVector();
        // Calculates a similarity score between the two vectors with a specified function. Higher similarity
        // scores correspond to closer vectors.

        // The query vector of Faiss byte vector is a Float array because ScalarQuantizer accepts it as float array.
        // To compute the score between this query vector and each vector in KNNByteVectorValues we are casting this query vector into byte
        // array directly.
        // This is safe to do so because float query vector already has validated byte values. Do not reuse this direct cast at any other
        // place.
        final byte[] byteQueryVector = new byte[queryVector.length];
        for (int i = 0; i < queryVector.length; i++) {
            byteQueryVector[i] = (byte) queryVector[i];
        }
        return spaceType.getKnnVectorSimilarityFunction().compare(byteQueryVector, vector);
    }

    protected int getNextDocId() throws IOException {
        if (bitSetIterator == null) {
            return byteVectorValues.nextDoc();
        }
        int nextDocID = this.bitSetIterator.nextDoc();
        // For filter case, advance vector values to corresponding doc id from filter bit set
        if (nextDocID != DocIdSetIterator.NO_MORE_DOCS) {
            byteVectorValues.advance(nextDocID);
        }
        return nextDocID;
    }
}

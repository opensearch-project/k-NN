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
import org.opensearch.knn.index.vectorvalues.KNNBinaryVectorValues;

import java.io.IOException;

/**
 * Inspired by DiversifyingChildrenFloatKnnVectorQuery in lucene
 * https://github.com/apache/lucene/blob/7b8aece125aabff2823626d5b939abf4747f63a7/lucene/join/src/java/org/apache/lucene/search/join/DiversifyingChildrenFloatKnnVectorQuery.java#L162
 *
 * The class is used in KNNWeight to score all docs, but, it iterates over filterIdsArray if filter is provided
 */
public class ByteVectorIdsKNNIterator implements KNNIterator {
    protected final BitSetIterator bitSetIterator;
    protected final byte[] queryVector;
    protected final KNNBinaryVectorValues binaryVectorValues;
    protected final SpaceType spaceType;
    protected float currentScore = Float.NEGATIVE_INFINITY;
    protected int docId;

    public ByteVectorIdsKNNIterator(
        @Nullable final BitSet filterIdsBitSet,
        final byte[] queryVector,
        final KNNBinaryVectorValues binaryVectorValues,
        final SpaceType spaceType
    ) throws IOException {
        this.bitSetIterator = filterIdsBitSet == null ? null : new BitSetIterator(filterIdsBitSet, filterIdsBitSet.length());
        this.queryVector = queryVector;
        this.binaryVectorValues = binaryVectorValues;
        this.spaceType = spaceType;
        this.docId = getNextDocId();
    }

    public ByteVectorIdsKNNIterator(final byte[] queryVector, final KNNBinaryVectorValues binaryVectorValues, final SpaceType spaceType)
        throws IOException {
        this(null, queryVector, binaryVectorValues, spaceType);
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
        if (bitSetIterator != null) {
            binaryVectorValues.advance(docId);
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
        final byte[] vector = binaryVectorValues.getVector();
        // Calculates a similarity score between the two vectors with a specified function. Higher similarity
        // scores correspond to closer vectors.
        return spaceType.getKnnVectorSimilarityFunction().compare(queryVector, vector);
    }

    protected int getNextDocId() throws IOException {
        if (bitSetIterator != null) {
            return bitSetIterator.nextDoc();
        }
        return binaryVectorValues.nextDoc();
    }
}

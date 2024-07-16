/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.filtered;

import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.util.KNNVectorSerializer;
import org.opensearch.knn.index.codec.util.KNNVectorSerializerFactory;

import java.io.IOException;

/**
 * Inspired by DiversifyingChildrenFloatKnnVectorQuery in lucene
 * https://github.com/apache/lucene/blob/7b8aece125aabff2823626d5b939abf4747f63a7/lucene/join/src/java/org/apache/lucene/search/join/DiversifyingChildrenFloatKnnVectorQuery.java#L162
 *
 * The class is used in KNNWeight to score filtered KNN field by iterating filterIdsArray.
 */
public class FilteredIdsKNNIterator implements KNNIterator {
    // Array of doc ids to iterate
    protected final BitSet filterIdsBitSet;
    protected final BitSetIterator bitSetIterator;
    protected final float[] queryVector;
    protected final BinaryDocValues binaryDocValues;
    protected final SpaceType spaceType;
    protected float currentScore = Float.NEGATIVE_INFINITY;
    protected int docId;

    public FilteredIdsKNNIterator(
        final BitSet filterIdsBitSet,
        final float[] queryVector,
        final BinaryDocValues binaryDocValues,
        final SpaceType spaceType
    ) {
        this.filterIdsBitSet = filterIdsBitSet;
        this.bitSetIterator = new BitSetIterator(filterIdsBitSet, filterIdsBitSet.length());
        this.queryVector = queryVector;
        this.binaryDocValues = binaryDocValues;
        this.spaceType = spaceType;
        this.docId = bitSetIterator.nextDoc();
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
        int doc = binaryDocValues.advance(docId);
        currentScore = computeScore();
        docId = bitSetIterator.nextDoc();
        return doc;
    }

    @Override
    public float score() {
        return currentScore;
    }

    protected float computeScore() throws IOException {
        final BytesRef value = binaryDocValues.binaryValue();
        final KNNVectorSerializer vectorSerializer = KNNVectorSerializerFactory.getSerializerByBytesRef(value);
        final float[] vector = vectorSerializer.byteToFloatArray(value);
        // Calculates a similarity score between the two vectors with a specified function. Higher similarity
        // scores correspond to closer vectors.
        return spaceType.getKnnVectorSimilarityFunction().compare(queryVector, vector);
    }
}

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
import org.apache.lucene.util.VectorUtil;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.util.KNNVectorSerializer;
import org.opensearch.knn.index.codec.util.KNNVectorSerializerFactory;
import org.opensearch.knn.index.util.KNNEngine;

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
    protected final KNNEngine knnEngine;
    protected float currentScore = Float.NEGATIVE_INFINITY;
    protected int docId;

    public FilteredIdsKNNIterator(
        final BitSet filterIdsBitSet,
        final float[] queryVector,
        final BinaryDocValues binaryDocValues,
        final SpaceType spaceType,
        final KNNEngine knnEngine
    ) {
        this.filterIdsBitSet = filterIdsBitSet;
        this.bitSetIterator = new BitSetIterator(filterIdsBitSet, filterIdsBitSet.length());
        this.queryVector = queryVector;
        this.binaryDocValues = binaryDocValues;
        this.spaceType = spaceType;
        this.knnEngine = knnEngine;
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
        // First uses the similarity function to calculate the raw score between the query vector and actual vector
        // After that we use KNNEngine.score method to convert the raw score to a valid document score.
        // In OpenSource, as Lucene is upgraded and we use MAX_INNER_PRODUCT, we don't use this kind of translation.
        // Ref: https://github.com/opensearch-project/k-NN/pull/1532
        // When we upgrade Lucene and OpenSearch version >=2.13 we should ensure that we switch the SpaceType.INNER_PRODUCT
        // to MAX_INNER_PRODUCT of Lucene and remove the below translation. Because in later versions of Lucene doing
        // VectorSimilarityFunction.compare actually returns the doc score(aka translated score) and not the raw score
        // Ref: https://github.com/opensearch-project/k-NN/pull/1532 for more details.
        if (spaceType == SpaceType.INNER_PRODUCT) {
            // only for dot product we need to ensure that we are getting raw score using Lucene functions.
            // This is a good fix till 2.11 version of OpenSearch. After this follow the above comment.
            return knnEngine.score(VectorUtil.dotProduct(queryVector, vector), spaceType);
        } else {
            return spaceType.getKnnVectorSimilarityFunction().compare(queryVector, vector);
        }
    }
}

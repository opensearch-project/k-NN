/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.scorers;

import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfFloatsSerializer;

import java.io.IOException;

/**
 * A {@link VectorScorer} backed by {@link BinaryDocValues}.
 *
 * <p>This scorer is used in the exact search path to score documents whose vectors are stored
 * as serialized bytes in {@link BinaryDocValues}. Document vectors are deserialized on the fly
 * and compared against the query vector using the similarity function derived from the
 * configured {@link SpaceType}.
 *
 * <p>Use the static factory methods to create an instance:
 * <ul>
 *   <li>{@link #create(float[], BinaryDocValues, SpaceType)}
 *       — for float[] query vectors. Document vectors are deserialized from {@link BytesRef} to
 *       float[] via {@link KNNVectorAsCollectionOfFloatsSerializer}.</li>
 *   <li>{@link #create(byte[], BinaryDocValues, SpaceType)}
 *       — for byte[] query vectors. Document vectors are extracted as raw byte[] from the
 *       {@link BytesRef}.</li>
 * </ul>
 */
public class KNNBinaryDocValuesScorer implements VectorScorer {

    private final BinaryDocValues binaryDocValues;
    private final ScoreFunction scoreFunction;

    /**
     * Strategy for computing a similarity score from a serialized document vector.
     *
     * <p>A functional interface is used here because the two factory methods ({@link #create(float[], BinaryDocValues, SpaceType)}
     * and {@link #create(byte[], BinaryDocValues, SpaceType)}) require different deserialization
     * and comparison logic. Each factory method captures its specific query vector type and
     * deserialization strategy in a lambda at construction time, avoiding runtime type checks
     * or branching in {@link #score()} on every call.
     */
    @FunctionalInterface
    private interface ScoreFunction {
        float score(BytesRef bytesRef) throws IOException;
    }

    private KNNBinaryDocValuesScorer(BinaryDocValues binaryDocValues, ScoreFunction scoreFunction) {
        this.binaryDocValues = binaryDocValues;
        this.scoreFunction = scoreFunction;
    }

    /**
     * Creates a scorer for a float[] query vector.
     *
     * @param queryVector        the query vector
     * @param binaryDocValues the binary doc values containing serialized document vectors
     * @param spaceType       the space type defining the similarity function
     * @return a new {@link KNNBinaryDocValuesScorer}
     */
    public static KNNBinaryDocValuesScorer create(float[] queryVector, BinaryDocValues binaryDocValues, SpaceType spaceType) {
        return new KNNBinaryDocValuesScorer(binaryDocValues, bytesRef -> {
            float[] docVector = KNNVectorAsCollectionOfFloatsSerializer.INSTANCE.byteToFloatArray(bytesRef);
            return spaceType.getKnnVectorSimilarityFunction().compare(queryVector, docVector);
        });
    }

    /**
     * Creates a scorer for a byte[] query vector.
     *
     * @param queryVector     the query vector
     * @param binaryDocValues the binary doc values containing serialized document vectors
     * @param spaceType       the space type defining the similarity function
     * @return a new {@link KNNBinaryDocValuesScorer}
     */
    public static KNNBinaryDocValuesScorer create(byte[] queryVector, BinaryDocValues binaryDocValues, SpaceType spaceType) {
        return new KNNBinaryDocValuesScorer(binaryDocValues, bytesRef -> {
            byte[] docVector = ArrayUtil.copyOfSubArray(bytesRef.bytes, bytesRef.offset, bytesRef.offset + bytesRef.length);
            return spaceType.getKnnVectorSimilarityFunction().compare(queryVector, docVector);
        });
    }

    @Override
    public float score() throws IOException {
        return scoreFunction.score(binaryDocValues.binaryValue());
    }

    @Override
    public DocIdSetIterator iterator() {
        return binaryDocValues;
    }
}

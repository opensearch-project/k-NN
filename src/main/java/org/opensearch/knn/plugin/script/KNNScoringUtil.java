/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import org.opensearch.knn.index.KNNVectorScriptDocValues;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.knn.index.VectorDataType;

import java.math.BigInteger;
import java.util.List;
import java.util.Objects;

import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.validateByteVectorValue;

public class KNNScoringUtil {
    private static Logger logger = LogManager.getLogger(KNNScoringUtil.class);

    /**
     * checks both query vector and input vector has equal dimension
     *
     * @param queryVector query vector
     * @param inputVector input vector
     * @throws IllegalArgumentException if query vector and input vector has different dimensions
     */
    private static void requireEqualDimension(final float[] queryVector, final float[] inputVector) {
        Objects.requireNonNull(queryVector);
        Objects.requireNonNull(inputVector);
        if (queryVector.length != inputVector.length) {
            String errorMessage = String.format(
                "query vector dimension mismatch. Expected: %d, Given: %d",
                inputVector.length,
                queryVector.length
            );
            throw new IllegalArgumentException(errorMessage);
        }
    }

    /**
     * This method calculates L2 squared distance between query vector
     * and input vector
     *
     * @param queryVector query vector
     * @param inputVector input vector
     * @return L2 score
     */
    public static float l2Squared(float[] queryVector, float[] inputVector) {
        requireEqualDimension(queryVector, inputVector);
        float squaredDistance = 0;
        for (int i = 0; i < inputVector.length; i++) {
            float diff = queryVector[i] - inputVector[i];
            squaredDistance += diff * diff;
        }
        return squaredDistance;
    }

    private static float[] toFloat(List<Number> inputVector, VectorDataType vectorDataType) {
        Objects.requireNonNull(inputVector);
        float[] value = new float[inputVector.size()];
        int index = 0;
        for (final Number val : inputVector) {
            float floatValue = val.floatValue();
            if (VectorDataType.BYTE == vectorDataType) {
                validateByteVectorValue(floatValue);
            }
            value[index++] = floatValue;
        }
        return value;
    }

    /**
     * Allowlisted l2Squared method for users to calculate L2 squared distance between query vector
     * and document vectors
     * Example
     *  "script": {
     *         "source": "1/(1 + l2Squared(params.query_vector, doc[params.field]))",
     *         "params": {
     *           "query_vector": [1, 2, 3.4],
     *           "field": "my_dense_vector"
     *         }
     *       }
     *
     * @param queryVector query vector
     * @param docValues   script doc values
     * @return L2 score
     */
    public static float l2Squared(List<Number> queryVector, KNNVectorScriptDocValues docValues) {
        return l2Squared(toFloat(queryVector, docValues.getVectorDataType()), docValues.getValue());
    }

    /**
     * This method can be used script to avoid repeated calculation of normalization
     * for query vector for each filtered documents
     *
     * @param queryVector     query vector
     * @param inputVector     input vector
     * @param normQueryVector normalized query vector value.
     * @return cosine score
     */
    public static float cosinesimilOptimized(float[] queryVector, float[] inputVector, float normQueryVector) {
        requireEqualDimension(queryVector, inputVector);
        float dotProduct = 0.0f;
        float normInputVector = 0.0f;
        for (int i = 0; i < queryVector.length; i++) {
            dotProduct += queryVector[i] * inputVector[i];
            normInputVector += inputVector[i] * inputVector[i];
        }
        float normalizedProduct = normQueryVector * normInputVector;
        if (normalizedProduct == 0) {
            logger.debug("Invalid vectors for cosine. Returning minimum score to put this result to end");
            return 0.0f;
        }
        return (float) (dotProduct / (Math.sqrt(normalizedProduct)));
    }

    /**
     * Allowlisted cosineSimilarity method that can be used in a script to avoid repeated
     * calculation of normalization for the query vector.
     * Example:
     *  "script": {
     *         "source": "cosineSimilarity(params.query_vector, docs[field], 1.0) ",
     *         "params": {
     *           "query_vector": [1, 2, 3.4],
     *           "field": "my_dense_vector"
     *         }
     *       }
     *
     * @param queryVector          query vector
     * @param docValues            script doc values
     * @param queryVectorMagnitude the magnitude of the query vector.
     * @return cosine score
     */
    public static float cosineSimilarity(List<Number> queryVector, KNNVectorScriptDocValues docValues, Number queryVectorMagnitude) {
        return cosinesimilOptimized(
            toFloat(queryVector, docValues.getVectorDataType()),
            docValues.getValue(),
            queryVectorMagnitude.floatValue()
        );
    }

    /**
     * This method calculates cosine similarity
     *
     * @param queryVector query vector
     * @param inputVector input vector
     * @return cosine score
     */
    public static float cosinesimil(float[] queryVector, float[] inputVector) {
        requireEqualDimension(queryVector, inputVector);
        float dotProduct = 0.0f;
        float normQueryVector = 0.0f;
        float normInputVector = 0.0f;
        for (int i = 0; i < queryVector.length; i++) {
            dotProduct += queryVector[i] * inputVector[i];
            normQueryVector += queryVector[i] * queryVector[i];
            normInputVector += inputVector[i] * inputVector[i];
        }
        float normalizedProduct = normQueryVector * normInputVector;
        if (normalizedProduct == 0) {
            logger.debug("Invalid vectors for cosine. Returning minimum score to put this result to end");
            return 0.0f;
        }
        return (float) (dotProduct / (Math.sqrt(normalizedProduct)));
    }

    /**
     * Allowlisted cosineSimilarity method for users to calculate cosine similarity between query vectors and
     * document vectors
     * Example:
     *  "script": {
     *         "source": "cosineSimilarity(params.query_vector, docs[field]) ",
     *         "params": {
     *           "query_vector": [1, 2, 3.4],
     *           "field": "my_dense_vector"
     *         }
     *       }
     *
     * @param queryVector query vector
     * @param docValues   script doc values
     * @return cosine score
     */
    public static float cosineSimilarity(List<Number> queryVector, KNNVectorScriptDocValues docValues) {
        return cosinesimil(toFloat(queryVector, docValues.getVectorDataType()), docValues.getValue());
    }

    /**
     * This method calculates hamming distance on 2 BigIntegers
     *
     * @param queryBigInteger BigInteger
     * @param inputBigInteger input BigInteger
     * @return hamming distance
     */
    public static float calculateHammingBit(BigInteger queryBigInteger, BigInteger inputBigInteger) {
        return inputBigInteger.xor(queryBigInteger).bitCount();
    }

    /**
     * This method calculates hamming distance on 2 longs
     *
     * @param queryLong query Long
     * @param inputLong input Long
     * @return hamming distance
     */
    public static float calculateHammingBit(Long queryLong, Long inputLong) {
        return Long.bitCount(queryLong ^ inputLong);
    }

    /**
     * This method calculates L1 distance between query vector
     * and input vector
     *
     * @param queryVector query vector
     * @param inputVector input vector
     * @return L1 score
     */
    public static float l1Norm(float[] queryVector, float[] inputVector) {
        requireEqualDimension(queryVector, inputVector);
        float distance = 0;
        for (int i = 0; i < inputVector.length; i++) {
            float diff = queryVector[i] - inputVector[i];
            distance += Math.abs(diff);
        }
        return distance;
    }

    /**
     * Allowlisted l1distance method for users to calculate L1 distance between query vector
     * and document vectors
     * Example
     *  "script": {
     *         "source": "1/(1 + l1Norm(params.query_vector, doc[params.field]))",
     *         "params": {
     *           "query_vector": [1, 2, 3.4],
     *           "field": "my_dense_vector"
     *         }
     *       }
     *
     * @param queryVector query vector
     * @param docValues   script doc values
     * @return L1 score
     */
    public static float l1Norm(List<Number> queryVector, KNNVectorScriptDocValues docValues) {
        return l1Norm(toFloat(queryVector, docValues.getVectorDataType()), docValues.getValue());
    }

    /**
     * This method calculates L-inf distance between query vector
     * and input vector
     *
     * @param queryVector query vector
     * @param inputVector input vector
     * @return L-inf score
     */
    public static float lInfNorm(float[] queryVector, float[] inputVector) {
        requireEqualDimension(queryVector, inputVector);
        float distance = 0;
        for (int i = 0; i < inputVector.length; i++) {
            float diff = queryVector[i] - inputVector[i];
            distance = Math.max(Math.abs(diff), distance);
        }
        return distance;
    }

    /**
     * Allowlisted lInfNorm method for users to calculate L-inf distance between query vector
     * and document vectors
     * Example
     *  "script": {
     *         "source": "1/(1 + lInfNorm(params.query_vector, doc[params.field]))",
     *         "params": {
     *           "query_vector": [1, 2, 3.4],
     *           "field": "my_dense_vector"
     *         }
     *       }
     *
     * @param queryVector query vector
     * @param docValues   script doc values
     * @return L-inf score
     */
    public static float lInfNorm(List<Number> queryVector, KNNVectorScriptDocValues docValues) {
        return lInfNorm(toFloat(queryVector, docValues.getVectorDataType()), docValues.getValue());
    }

    /**
     * This method calculates dot product distance between query vector
     * and input vector
     *
     * @param queryVector query vector
     * @param inputVector input vector
     * @return dot product score
     */
    public static float innerProduct(float[] queryVector, float[] inputVector) {
        requireEqualDimension(queryVector, inputVector);
        float distance = 0;
        for (int i = 0; i < inputVector.length; i++) {
            distance += queryVector[i] * inputVector[i];
        }
        return distance;
    }

    /**
     * Allowlisted innerProd method for users to calculate inner product distance between query vector
     * and document vectors
     * Example
     *  "script": {
     *         "source": "float x = innerProduct([1.0f, 1.0f], doc['%s']); return x&gt;=0? 2-(1/(x+1)):1/(1-x);",
     *         "params": {
     *           "query_vector": [1, 2, 3.4],
     *           "field": "my_dense_vector"
     *         }
     *       }
     *
     * @param queryVector query vector
     * @param docValues   script doc values
     * @return inner product score
     */
    public static float innerProduct(List<Number> queryVector, KNNVectorScriptDocValues docValues) {
        return innerProduct(toFloat(queryVector, docValues.getVectorDataType()), docValues.getValue());
    }
}

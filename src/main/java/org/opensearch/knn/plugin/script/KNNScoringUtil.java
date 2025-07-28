/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import java.math.BigInteger;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.util.VectorUtil;
import org.opensearch.knn.index.KNNVectorScriptDocValues;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;

import static org.opensearch.knn.common.KNNValidationUtil.validateByteVectorValue;

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
     * checks both query vector and input vector has equal dimension
     *
     * @param queryVector query vector
     * @param inputVector input vector
     * @throws IllegalArgumentException if query vector and input vector has different dimensions
     */
    private static void requireEqualDimension(final byte[] queryVector, final byte[] inputVector) {
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

    private static void requireNonBinaryType(final String spaceName, final VectorDataType vectorDataType) {
        if (VectorDataType.BINARY == vectorDataType) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "Incompatible field_type for %s space. The data type should be either float or byte but got %s",
                    spaceName,
                    vectorDataType.getValue()
                )
            );
        }
    }

    private static void requireBinaryType(final String spaceName, final VectorDataType vectorDataType) {
        if (VectorDataType.BINARY != vectorDataType) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "Incompatible field_type for %s space. The data type should be binary but got %s",
                    spaceName,
                    vectorDataType.getValue()
                )
            );
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
        return VectorUtil.squareDistance(queryVector, inputVector);
    }

    /**
     * This method calculates L2 squared distance between byte query vector
     * and byte input vector
     *
     * @param queryVector byte query vector
     * @param inputVector byte input vector
     * @return L2 score
     */
    public static float l2Squared(byte[] queryVector, byte[] inputVector) {
        return VectorUtil.squareDistance(queryVector, inputVector);
    }

    /**
     * Calculates the L2 squared distance between a float query vector and a binary document vector using ADC (Asymmetric Distance Computation).
     * This method implements a specialized version of L2 distance calculation where one vector is in binary format (compressed)
     * and the other is in float format (uncompressed).
     *
     * TODO: For now this will be very inefficient. We can in the future optimize through the following:
     * Use FloatVector.SPECIES_PREFERRED for SIMD processing in chunks with reduceLanes(),
     * Configure build.gradle with --add-modules jdk.incubator.vector --enable-preview flags into the VectorUtils class.
     *
     * @param queryVector The uncompressed query vector in float format
     * @param inputVector The compressed document vector in binary format, where each bit represents a dimension
     * @return The L2 squared distance between the two vectors. Lower values indicate closer vectors.
     * @throws IllegalArgumentException if queryVector length is not compatible with inputVector length (queryVector.length != inputVector.length * 8)
     */
    public static float l2SquaredADC(float[] queryVector, byte[] inputVector) {
        // we cannot defer to VectorUtil as it does not support ADC.
        float score = 0;

        for (int i = 0; i < queryVector.length; ++i) {
            int byteIndex = i / 8;
            int bitOffset = 7 - (i % 8);
            int bitValue = (inputVector[byteIndex] & (1 << bitOffset)) != 0 ? 1 : 0;

            // Calculate squared difference
            float diff = bitValue - queryVector[i];
            score += diff * diff;
        }
        return score;
    }

    /**
     * Calculates the inner product similarity between a float query vector and a binary document vector using ADC
     * (Asymmetric Distance Computation). This method is useful for similarity searches where one vector is compressed
     * in binary format and the other remains in float format.
     *
     * The inner product is calculated by summing the products of corresponding elements, where the binary vector's
     * elements are interpreted as 0 or 1.
     *
     * TODO: For now this will be very inefficient. We can in the future optimize through the following:
     * Use FloatVector.SPECIES_PREFERRED for SIMD processing in chunks with reduceLanes(),
     * Configure build.gradle with --add-modules jdk.incubator.vector --enable-preview flags into the VectorUtils class.
     *
     * @param queryVector The uncompressed query vector in float format
     * @param inputVector The compressed document vector in binary format, where each bit represents a dimension
     * @return The inner product similarity score between the two vectors. Higher values indicate more similar vectors.
     * @throws IllegalArgumentException if queryVector length is not compatible with inputVector length (queryVector.length != inputVector.length * 8)
     */
    public static float innerProductADC(float[] queryVector, byte[] inputVector) {
        float score = 0;

        for (int i = 0; i < queryVector.length; ++i) {
            // Extract the bit for this dimension
            int byteIndex = i / 8;
            int bitOffset = 7 - (i % 8);
            int bitValue = (inputVector[byteIndex] & (1 << bitOffset)) != 0 ? 1 : 0;

            // Calculate product and accumulate
            score += bitValue * queryVector[i];
        }
        return score;
    }

    private static float[] toFloat(final List<Number> inputVector, final VectorDataType vectorDataType) {
        Objects.requireNonNull(inputVector);
        float[] value = new float[inputVector.size()];
        int index = 0;
        for (final Number val : inputVector) {
            float floatValue = val.floatValue();
            if (VectorDataType.BYTE == vectorDataType || VectorDataType.BINARY == vectorDataType) {
                validateByteVectorValue(floatValue, vectorDataType);
            }
            value[index++] = floatValue;
        }
        return value;
    }

    private static byte[] toByte(final List<Number> inputVector, final VectorDataType vectorDataType) {
        Objects.requireNonNull(inputVector);
        byte[] value = new byte[inputVector.size()];
        int index = 0;
        for (final Number val : inputVector) {
            float floatValue = val.floatValue();
            if (VectorDataType.BYTE == vectorDataType || VectorDataType.BINARY == vectorDataType) {
                validateByteVectorValue(floatValue, vectorDataType);
            }
            value[index++] = val.byteValue();
        }
        return value;
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
        try {
            return VectorUtil.cosine(queryVector, inputVector);
        } catch (IllegalArgumentException | AssertionError e) {
            logger.debug("Invalid vectors for cosine. Returning minimum score to put this result to end");
            return 0.0f;
        }
    }

    /**
     * This method calculates cosine similarity
     *
     * @param queryVector byte query vector
     * @param inputVector byte input vector
     * @return cosine score
     */
    public static float cosinesimil(byte[] queryVector, byte[] inputVector) {
        requireEqualDimension(queryVector, inputVector);
        try {
            return VectorUtil.cosine(queryVector, inputVector);
        } catch (IllegalArgumentException | AssertionError e) {
            logger.debug("Invalid vectors for cosine. Returning minimum score to put this result to end");
            return 0.0f;
        }
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
     * This method calculates hamming distance between query vector
     * and input vector
     *
     * @param queryVector query vector
     * @param inputVector input vector
     * @return hamming distance
     */
    public static float calculateHammingBit(byte[] queryVector, byte[] inputVector) {
        requireEqualDimension(queryVector, inputVector);
        return VectorUtil.xorBitCount(queryVector, inputVector);
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
     * This method calculates L1 distance between byte query vector
     * and byte input vector
     *
     * @param queryVector byte query vector
     * @param inputVector byte input vector
     * @return L1 score
     */
    public static float l1Norm(byte[] queryVector, byte[] inputVector) {
        requireEqualDimension(queryVector, inputVector);
        float distance = 0;
        for (int i = 0; i < inputVector.length; i++) {
            float diff = queryVector[i] - inputVector[i];
            distance += Math.abs(diff);
        }
        return distance;
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
     * This method calculates L-inf distance between byte query vector
     * and input vector
     *
     * @param queryVector byte query vector
     * @param inputVector byte input vector
     * @return L-inf score
     */
    public static float lInfNorm(byte[] queryVector, byte[] inputVector) {
        requireEqualDimension(queryVector, inputVector);
        float distance = 0;
        for (int i = 0; i < inputVector.length; i++) {
            float diff = queryVector[i] - inputVector[i];
            distance = Math.max(Math.abs(diff), distance);
        }
        return distance;
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
        return VectorUtil.dotProduct(queryVector, inputVector);
    }

    /**
     * This method calculates dot product distance between byte query vector
     * and byte input vector
     *
     * @param queryVector query vector
     * @param inputVector input vector
     * @return dot product score
     */
    public static float innerProduct(byte[] queryVector, byte[] inputVector) {
        requireEqualDimension(queryVector, inputVector);
        return VectorUtil.dotProduct(queryVector, inputVector);
    }

    /**
     *********************************************************************************************
     * Functions to be used in painless script which is defined in knn_allowlist.txt
     *********************************************************************************************
     */

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
    public static float l2Squared(List<Number> queryVector, KNNVectorScriptDocValues<?> docValues) {
        final VectorDataType vectorDataType = docValues.getVectorDataType();
        requireNonBinaryType("l2Squared", vectorDataType);
        if (VectorDataType.FLOAT == vectorDataType || VectorDataType.HALF_FLOAT == vectorDataType) {
            return l2Squared(toFloat(queryVector, docValues.getVectorDataType()), (float[]) docValues.getValue());
        }
        return l2Squared(toByte(queryVector, docValues.getVectorDataType()), (byte[]) docValues.getValue());
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
    public static float lInfNorm(List<Number> queryVector, KNNVectorScriptDocValues<?> docValues) {
        final VectorDataType vectorDataType = docValues.getVectorDataType();
        requireNonBinaryType("lInfNorm", vectorDataType);
        if (VectorDataType.FLOAT == vectorDataType || VectorDataType.HALF_FLOAT == vectorDataType) {
            return lInfNorm(toFloat(queryVector, docValues.getVectorDataType()), (float[]) docValues.getValue());
        }
        return lInfNorm(toByte(queryVector, docValues.getVectorDataType()), (byte[]) docValues.getValue());
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
    public static float l1Norm(List<Number> queryVector, KNNVectorScriptDocValues<?> docValues) {
        final VectorDataType vectorDataType = docValues.getVectorDataType();
        requireNonBinaryType("l1Norm", vectorDataType);
        if (VectorDataType.FLOAT == vectorDataType || VectorDataType.HALF_FLOAT == vectorDataType) {
            return l1Norm(toFloat(queryVector, docValues.getVectorDataType()), (float[]) docValues.getValue());
        }
        return l1Norm(toByte(queryVector, docValues.getVectorDataType()), (byte[]) docValues.getValue());
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
    public static float innerProduct(List<Number> queryVector, KNNVectorScriptDocValues<?> docValues) {
        final VectorDataType vectorDataType = docValues.getVectorDataType();
        requireNonBinaryType("innerProduct", vectorDataType);
        if (VectorDataType.FLOAT == vectorDataType || VectorDataType.HALF_FLOAT == vectorDataType) {
            return innerProduct(toFloat(queryVector, docValues.getVectorDataType()), (float[]) docValues.getValue());
        }
        return innerProduct(toByte(queryVector, docValues.getVectorDataType()), (byte[]) docValues.getValue());
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
    public static float cosineSimilarity(List<Number> queryVector, KNNVectorScriptDocValues<?> docValues) {
        final VectorDataType vectorDataType = docValues.getVectorDataType();
        requireNonBinaryType("cosineSimilarity", vectorDataType);
        if (VectorDataType.FLOAT == vectorDataType || VectorDataType.HALF_FLOAT == vectorDataType) {
            float[] inputVector = toFloat(queryVector, docValues.getVectorDataType());
            SpaceType.COSINESIMIL.validateVector(inputVector);
            return cosinesimil(inputVector, (float[]) docValues.getValue());
        } else {
            byte[] inputVector = toByte(queryVector, docValues.getVectorDataType());
            SpaceType.COSINESIMIL.validateVector(inputVector);
            return cosinesimil(inputVector, (byte[]) docValues.getValue());
        }
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
    public static float cosineSimilarity(List<Number> queryVector, KNNVectorScriptDocValues<?> docValues, Number queryVectorMagnitude) {
        final VectorDataType vectorDataType = docValues.getVectorDataType();
        requireNonBinaryType("cosineSimilarity", vectorDataType);
        float[] inputVector = toFloat(queryVector, docValues.getVectorDataType());
        SpaceType.COSINESIMIL.validateVector(inputVector);
        if (VectorDataType.FLOAT == vectorDataType || VectorDataType.HALF_FLOAT == vectorDataType) {
            return cosinesimilOptimized(inputVector, (float[]) docValues.getValue(), queryVectorMagnitude.floatValue());
        } else {
            byte[] docVectorInByte = (byte[]) docValues.getValue();
            float[] docVectorInFloat = new float[docVectorInByte.length];
            for (int i = 0; i < docVectorInByte.length; i++) {
                docVectorInFloat[i] = docVectorInByte[i];
            }
            return cosinesimilOptimized(inputVector, docVectorInFloat, queryVectorMagnitude.floatValue());
        }
    }

    /**
     * Allowlisted hamming method that can be used in a script to avoid repeated
     * calculation of normalization for the query vector.
     * Example:
     *  "script": {
     *         "source": "hamming(params.query_vector, docs[field]) ",
     *         "params": {
     *           "query_vector": [1, 2],
     *           "field": "my_dense_vector"
     *         }
     *       }
     *
     * @param queryVector          query vector
     * @param docValues            script doc values
     * @return hamming score
     */
    public static float hamming(List<Number> queryVector, KNNVectorScriptDocValues<?> docValues) {
        requireBinaryType("hamming", docValues.getVectorDataType());
        byte[] queryVectorInByte = toByte(queryVector, docValues.getVectorDataType());
        return calculateHammingBit(queryVectorInByte, (byte[]) docValues.getValue());
    }
}

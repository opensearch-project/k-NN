/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index;

import lombok.Getter;

import java.util.Locale;

import java.util.HashSet;
import java.util.Set;

import static org.opensearch.knn.common.KNNVectorUtil.isZeroVector;

/**
 * Enum contains spaces supported for approximate nearest neighbor search in the k-NN plugin. Each engine's methods are
 * expected to support a subset of these spaces. Validation should be done in the jni layer and an exception should be
 * propagated up to the Java layer. Additionally, naming translations should be done in jni layer as well. For example,
 * nmslib calls the inner_product space "negdotprod". This translation should take place in the nmslib's jni layer.
 */
@Getter
public enum SpaceType {
    L2("l2") {
        @Override
        public float scoreTranslation(float rawScore) {
            return 1 / (1 + rawScore);
        }

        @Override
        public KNNVectorSimilarityFunction getKnnVectorSimilarityFunction() {
            return KNNVectorSimilarityFunction.EUCLIDEAN;
        }

        @Override
        public float scoreToDistanceTranslation(float score) {
            if (score == 0) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "score cannot be 0 when space type is [%s]", getValue()));
            }
            return 1 / score - 1;
        }
    },
    COSINESIMIL("cosinesimil") {
        @Override
        public float scoreTranslation(float rawScore) {
            return 1 / (1 + rawScore);
        }

        @Override
        public KNNVectorSimilarityFunction getKnnVectorSimilarityFunction() {
            return KNNVectorSimilarityFunction.COSINE;
        }

        @Override
        public void validateVector(byte[] vector) {
            if (isZeroVector(vector)) {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "zero vector is not supported when space type is [%s]", getValue())
                );
            }
        }

        @Override
        public void validateVector(float[] vector) {
            if (isZeroVector(vector)) {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "zero vector is not supported when space type is [%s]", getValue())
                );
            }
        }
    },
    L1("l1") {
        @Override
        public float scoreTranslation(float rawScore) {
            return 1 / (1 + rawScore);
        }
    },
    LINF("linf") {
        @Override
        public float scoreTranslation(float rawScore) {
            return 1 / (1 + rawScore);
        }
    },
    INNER_PRODUCT("innerproduct") {
        @Override
        public float scoreTranslation(float rawScore) {
            if (rawScore >= 0) {
                return 1 / (1 + rawScore);
            }
            return -rawScore + 1;
        }

        @Override
        public KNNVectorSimilarityFunction getKnnVectorSimilarityFunction() {
            return KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;
        }
    },
    HAMMING("hamming") {
        @Override
        public float scoreTranslation(float rawScore) {
            return 1 / (1 + rawScore);
        }

        @Override
        public void validateVectorDataType(VectorDataType vectorDataType) {
            if (VectorDataType.BINARY != vectorDataType) {
                throw new IllegalArgumentException(
                    String.format(
                        Locale.ROOT,
                        "Space type [%s] is not supported with [%s] data type",
                        getValue(),
                        vectorDataType.getValue()
                    )
                );
            }
        }

        @Override
        public KNNVectorSimilarityFunction getKnnVectorSimilarityFunction() {
            return KNNVectorSimilarityFunction.HAMMING;
        }
    };

    public static SpaceType DEFAULT = L2;
    public static SpaceType DEFAULT_BINARY = HAMMING;

    private final String value;

    SpaceType(String value) {
        this.value = value;
    }

    public abstract float scoreTranslation(float rawScore);

    /**
     * Get KNNVectorSimilarityFunction that maps to this SpaceType
     *
     * @return KNNVectorSimilarityFunction
     */
    public KNNVectorSimilarityFunction getKnnVectorSimilarityFunction() {
        throw new UnsupportedOperationException(String.format("Space [%s] does not have a knn vector similarity function", getValue()));
    }

    /**
     * Validate if the given byte vector is supported by this space type
     *
     * @param vector     the given vector
     */
    public void validateVector(byte[] vector) {
        // do nothing
    }

    /**
     * Validate if the given float vector is supported by this space type
     *
     * @param vector     the given vector
     */
    public void validateVector(float[] vector) {
        // do nothing
    }

    /**
     * Validate if given vector data type is supported by this space type
     *
     * @param vectorDataType the given vector data type
     */
    public void validateVectorDataType(VectorDataType vectorDataType) {
        if (VectorDataType.FLOAT != vectorDataType && VectorDataType.BYTE != vectorDataType) {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "Space type [%s] is not supported with [%s] data type", getValue(), vectorDataType.getValue())
            );
        }
    }

    public static Set<String> getValues() {
        Set<String> values = new HashSet<>();

        for (SpaceType spaceType : SpaceType.values()) {
            values.add(spaceType.getValue());
        }
        return values;
    }

    public static SpaceType getSpace(String spaceTypeName) {
        for (SpaceType currentSpaceType : SpaceType.values()) {
            if (currentSpaceType.getValue().equalsIgnoreCase(spaceTypeName)) {
                return currentSpaceType;
            }
        }
        throw new IllegalArgumentException("Unable to find space: " + spaceTypeName);
    }

    /**
     * Translate a score to a distance for this space type
     *
     * @param score score to translate
     * @return translated distance
     */
    public float scoreToDistanceTranslation(float score) {
        throw new UnsupportedOperationException(String.format("Space [%s] does not have a score to distance translation", getValue()));
    }
}

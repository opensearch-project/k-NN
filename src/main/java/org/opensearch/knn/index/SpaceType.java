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

import java.util.Locale;
import org.apache.lucene.index.VectorSimilarityFunction;

import java.util.HashSet;
import java.util.Set;

import static org.opensearch.knn.common.KNNVectorUtil.isZeroVector;

/**
 * Enum contains spaces supported for approximate nearest neighbor search in the k-NN plugin. Each engine's methods are
 * expected to support a subset of these spaces. Validation should be done in the jni layer and an exception should be
 * propagated up to the Java layer. Additionally, naming translations should be done in jni layer as well. For example,
 * nmslib calls the inner_product space "negdotprod". This translation should take place in the nmslib's jni layer.
 */
public enum SpaceType {
    L2("l2") {
        @Override
        public float scoreTranslation(float rawScore) {
            return 1 / (1 + rawScore);
        }

        @Override
        public VectorSimilarityFunction getVectorSimilarityFunction() {
            return VectorSimilarityFunction.EUCLIDEAN;
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
        public VectorSimilarityFunction getVectorSimilarityFunction() {
            return VectorSimilarityFunction.COSINE;
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
        /**
         * The inner product has a range of [-Float.MAX_VALUE, Float.MAX_VALUE], with a more similar result being
         * represented by a more negative value. In Lucene, scores have to be in the range of [0, Float.MAX_VALUE],
         * where a higher score represents a more similar result. So, we convert here.
         *
         * @param rawScore score returned from underlying library
         * @return Lucene scaled score
         */
        @Override
        public float scoreTranslation(float rawScore) {
            if (rawScore >= 0) {
                return 1 / (1 + rawScore);
            }
            return -rawScore + 1;
        }

        @Override
        public VectorSimilarityFunction getVectorSimilarityFunction() {
            return VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;
        }
    },
    HAMMING_BIT("hammingbit") {
        @Override
        public float scoreTranslation(float rawScore) {
            return 1 / (1 + rawScore);
        }
    };

    public static SpaceType DEFAULT = L2;

    private final String value;

    SpaceType(String value) {
        this.value = value;
    }

    public abstract float scoreTranslation(float rawScore);

    /**
     * Get VectorSimilarityFunction that maps to this SpaceType
     *
     * @return VectorSimilarityFunction
     */
    public VectorSimilarityFunction getVectorSimilarityFunction() {
        throw new UnsupportedOperationException(String.format("Space [%s] does not have a vector similarity function", getValue()));
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
     * Get space type name in engine
     *
     * @return name
     */
    public String getValue() {
        return value;
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

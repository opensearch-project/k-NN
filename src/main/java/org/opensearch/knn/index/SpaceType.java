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

import org.apache.lucene.index.VectorSimilarityFunction;

import java.util.Arrays;
import java.util.Locale;

import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNVectorUtil.isZeroVector;

/**
 * Enum contains spaces supported for approximate nearest neighbor search in the k-NN plugin. Each engine's methods are
 * expected to support a subset of these spaces. Validation should be done in the jni layer and an exception should be
 * propagated up to the Java layer. Additionally, naming translations should be done in jni layer as well. For example,
 * nmslib calls the inner_product space "negdotprod". This translation should take place in the nmslib's jni layer.
 */
public enum SpaceType {
    // This undefined space type is used to indicate that space type is not provided by user
    // Later, we need to assign a default value based on data type
    UNDEFINED("undefined") {
        @Override
        public float scoreTranslation(final float rawScore) {
            throw new IllegalStateException("Unsupported method");
        }

        @Override
        public KNNVectorSimilarityFunction getKnnVectorSimilarityFunction() {
            // not supported
            return null;
        }

        @Override
        public void validateVectorDataType(VectorDataType vectorDataType) {
            throw new IllegalStateException("Unsupported method");
        }
    },
    L2("l2", SpaceType.GENERIC_SCORE_TRANSLATION) {
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
    COSINESIMIL("cosinesimil", "`Math.max((2.0F - rawScore) / 2.0F, 0.0F)`") {
        /**
         * Cosine similarity has range of [-1, 1] where -1 represents vectors are at diametrically opposite, and 1 is where
         * they are identical in direction and perfectly similar. In Lucene, scores have to be in the range of [0, Float.MAX_VALUE].
         * Hence, to move the range from [-1, 1] to [ 0, Float.MAX_VALUE], we convert  using following formula which is adopted
         * by Lucene as mentioned here
         * https://github.com/apache/lucene/blob/0494c824e0ac8049b757582f60d085932a890800/lucene/core/src/java/org/apache/lucene/index/VectorSimilarityFunction.java#L73
         * We expect raw score = 1 - cosine(x,y), if underlying library returns different range or other than expected raw score,
         * they should override this method to either provide valid range or convert raw score to the format as 1 - cosine and call this method
         *
         * @param rawScore score returned from underlying library
         * @return Lucene scaled score
         */
        @Override
        public float scoreTranslation(float rawScore) {
            return Math.max((2.0F - rawScore) / 2.0F, 0.0F);
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
    L1("l1", SpaceType.GENERIC_SCORE_TRANSLATION) {
        @Override
        public float scoreTranslation(float rawScore) {
            return 1 / (1 + rawScore);
        }

        @Override
        public KNNVectorSimilarityFunction getKnnVectorSimilarityFunction() {
            // not supported
            return null;
        }
    },
    LINF("linf", SpaceType.GENERIC_SCORE_TRANSLATION) {
        @Override
        public float scoreTranslation(float rawScore) {
            return 1 / (1 + rawScore);
        }

        @Override
        public KNNVectorSimilarityFunction getKnnVectorSimilarityFunction() {
            return null;
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
        public String explainScoreTranslation(float rawScore) {
            return rawScore >= 0 ? GENERIC_SCORE_TRANSLATION : "`-rawScore + 1`";
        }

        @Override
        public KNNVectorSimilarityFunction getKnnVectorSimilarityFunction() {
            return KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;
        }
    },
    HAMMING("hamming", SpaceType.GENERIC_SCORE_TRANSLATION) {
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

    private static final String[] VALID_VALUES = Arrays.stream(SpaceType.values())
        .filter(space -> space != SpaceType.UNDEFINED)
        .map(SpaceType::getValue)
        .collect(Collectors.toList())
        .toArray(new String[0]);

    private static final String GENERIC_SCORE_TRANSLATION = "`1 / (1 + rawScore)`";
    private final String value;
    private final String explanationFormula;

    SpaceType(String value) {
        this.value = value;
        this.explanationFormula = null;
    }

    SpaceType(String value, String explanationFormula) {
        this.value = value;
        this.explanationFormula = explanationFormula;
    }

    public abstract float scoreTranslation(float rawScore);

    public String explainScoreTranslation(float rawScore) {
        if (explanationFormula != null) {
            return explanationFormula;
        }
        throw new UnsupportedOperationException("explainScoreTranslation is not defined for this space type.");
    }

    /**
     * Get KNNVectorSimilarityFunction that maps to this SpaceType
     *
     * @return KNNVectorSimilarityFunction
     */
    public abstract KNNVectorSimilarityFunction getKnnVectorSimilarityFunction();

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
        if (VectorDataType.FLOAT != vectorDataType
            && VectorDataType.HALF_FLOAT != vectorDataType
            && VectorDataType.BYTE != vectorDataType) {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "Space type [%s] is not supported with [%s] data type", getValue(), vectorDataType.getValue())
            );
        }
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
        throw new IllegalArgumentException(
            String.format(Locale.ROOT, "Unable to find space: %s . Valid values are: %s", spaceTypeName, Arrays.toString(VALID_VALUES))
        );
    }

    public static SpaceType getSpace(VectorSimilarityFunction similarityFunction) {
        for (SpaceType currentSpaceType : SpaceType.values()) {
            KNNVectorSimilarityFunction knnSimilarityFunction = currentSpaceType.getKnnVectorSimilarityFunction();
            if (knnSimilarityFunction != null && knnSimilarityFunction.getVectorSimilarityFunction() == similarityFunction) {
                return currentSpaceType;
            }
        }
        throw new IllegalArgumentException(
            String.format(
                Locale.ROOT,
                "Unable to find space type for similarity function : %s . Valid values are: %s",
                similarityFunction,
                Arrays.toString(KNNVectorSimilarityFunction.values())
            )
        );
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

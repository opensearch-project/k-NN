/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

/**
 * Enum for statistical operators used in KNN vector computation.
 */
public enum StatisticalOperators implements Computation {
    MEAN {
        @Override
        public float[] apply(float a, float b) {
            return new float[] { a + b };
        }

        @Override
        public float[] apply(float[] sum, long count) {
            float[] result = new float[sum.length];
            for (int i = 0; i < sum.length; i++) {
                result[i] = sum[i] / count;
            }
            return result;
        }
    },
    VARIANCE {
        @Override
        public float[] apply(float a, float b) {
            float diff = b - a;
            return new float[] { a + (diff * diff) };
        }

        @Override
        public float[] apply(float[] sum, long count) {
            float[] result = new float[sum.length];
            for (int i = 0; i < sum.length; i++) {
                result[i] = sum[i] / (count - 1);
            }
            return result;
        }
    },
    STANDARD_DEVIATION {
        @Override
        public float[] apply(float a, float b) {
            return VARIANCE.apply(a, b);
        }

        @Override
        public float[] apply(float[] sum, long count) {
            float[] variance = VARIANCE.apply(sum, count);
            float[] result = new float[variance.length];
            for (int i = 0; i < variance.length; i++) {
                result[i] = (float) Math.sqrt(variance[i]);
            }
            return result;
        }
    };
}
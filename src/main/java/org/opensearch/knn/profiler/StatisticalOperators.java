/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profiler;

/**
 * Enum for statistical operators used in KNN vector computation.
 */
public enum StatisticalOperators implements Computation {
    MEAN {
        @Override
        public float[] compute(float[] perDimensionVector) {
            if (perDimensionVector == null || perDimensionVector.length == 0) {
                return new float[] { 0.0f };
            }

            float sum = 0.0f;
            for (float value : perDimensionVector) {
                sum += value;
            }
            return new float[] { sum / perDimensionVector.length };
        }
    },
    VARIANCE {
        @Override
        public float[] compute(float[] perDimensionVector) {
            if (perDimensionVector == null || perDimensionVector.length <= 1) {
                return new float[] { 0.0f };
            }

            // First calculate mean
            float mean = MEAN.compute(perDimensionVector)[0];

            // Calculate sum of squared differences from mean
            float sumSquaredDiff = 0.0f;
            for (float value : perDimensionVector) {
                float diff = value - mean;
                sumSquaredDiff += diff * diff;
            }

            // Return sample variance
            return new float[] { sumSquaredDiff / (perDimensionVector.length - 1) };
        }
    },
    STANDARD_DEVIATION {
        @Override
        public float[] compute(float[] perDimensionVector) {
            float[] variance = VARIANCE.compute(perDimensionVector);
            return new float[] { (float) Math.sqrt(variance[0]) };
        }
    };
}

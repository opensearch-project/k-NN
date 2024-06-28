/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.knn.plugin.script.KNNScoringUtil;

/**
 * Wrapper class of VectorSimilarityFunction to support more function than what Lucene provides
 */
public enum KNNVectorSimilarityFunction {
    EUCLIDEAN {
        public float compare(float[] v1, float[] v2) {
            return VectorSimilarityFunction.EUCLIDEAN.compare(v1, v2);
        }

        public float compare(byte[] v1, byte[] v2) {
            return VectorSimilarityFunction.EUCLIDEAN.compare(v1, v2);
        }
    },
    DOT_PRODUCT {
        public float compare(float[] v1, float[] v2) {
            return VectorSimilarityFunction.DOT_PRODUCT.compare(v1, v2);
        }

        public float compare(byte[] v1, byte[] v2) {
            return VectorSimilarityFunction.DOT_PRODUCT.compare(v1, v2);
        }
    },
    COSINE {
        public float compare(float[] v1, float[] v2) {
            return VectorSimilarityFunction.COSINE.compare(v1, v2);
        }

        public float compare(byte[] v1, byte[] v2) {
            return VectorSimilarityFunction.COSINE.compare(v1, v2);
        }
    },
    MAXIMUM_INNER_PRODUCT {
        public float compare(float[] v1, float[] v2) {
            return VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT.compare(v1, v2);
        }

        public float compare(byte[] v1, byte[] v2) {
            return VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT.compare(v1, v2);
        }
    },
    HAMMING {
        public float compare(float[] v1, float[] v2) {
            throw new IllegalStateException("Hamming space is not supported with float vectors");
        }

        public float compare(byte[] v1, byte[] v2) {
            return 1.0f / (1 + KNNScoringUtil.calculateHammingBit(v1, v2));
        }
    };

    private KNNVectorSimilarityFunction() {}

    public abstract float compare(float[] var1, float[] var2);

    public abstract float compare(byte[] var1, byte[] var2);
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.util.VectorUtil;

public enum KNNVectorDistanceFunction {
    EUCLIDEAN {
        @Override
        public float distance(float[] vec1, float[] vec2) {
            return VectorUtil.squareDistance(vec1, vec2);
        }

        @Override
        public float distance(byte[] vec1, byte[] vec2) {
            return VectorUtil.squareDistance(vec1, vec2);
        }
    },
    DOT_PRODUCT {
        @Override
        public float distance(float[] vec1, float[] vec2) {
            return VectorUtil.dotProduct(vec1, vec2);
        }

        @Override
        public float distance(byte[] vec1, byte[] vec2) {
            return VectorUtil.dotProduct(vec1, vec2);
        }
    },
    COSINE {
        @Override
        public float distance(float[] vec1, float[] vec2) {
            return VectorUtil.cosine(vec1, vec2);
        }

        @Override
        public float distance(byte[] vec1, byte[] vec2) {
            return VectorUtil.cosine(vec1, vec2);
        }
    };

    public abstract float distance(float[] vec1, float[] vec2);

    public abstract float distance(byte[] vec1, byte[] vec2);
}

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
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.util.KNNEngine;

import java.util.Arrays;
import java.util.List;

public class SpaceTypeTests extends KNNTestCase {

    public void testGetVectorSimilarityFunction_l2() {
        assertEquals(VectorSimilarityFunction.EUCLIDEAN, SpaceType.L2.getVectorSimilarityFunction());
    }

    public void testGetVectorSimilarityFunction_invalid() {
        expectThrows(UnsupportedOperationException.class, SpaceType.L1::getVectorSimilarityFunction);
    }

    public void testGetVectorSimilarityFunction_whenInnerproduct_thenConsistentWithScoreTranslation() {
        /*
            For the innerproduct space type, we expect that negative dot product scores will be transformed as follows:
                if (negativeDotProduct >= 0) {
                    return 1 / (1 + negativeDotProduct);
                }
                return -negativeDotProduct + 1;

            Internally, Lucene uses scaleMaxInnerProductScore to scale the raw dot product into a proper lucene score.
            See:
                1. https://github.com/apache/lucene/blob/releases/lucene/9.10.0/lucene/core/src/java/org/apache/lucene/util/VectorUtil.java#L195-L200
                2. https://github.com/apache/lucene/blob/releases/lucene/9.10.0/lucene/core/src/java/org/apache/lucene/index/VectorSimilarityFunction.java#L90
         */
        final List<float[]> dataVectors = Arrays.asList(
            new float[] { 0.0f, 0.0f },
            new float[] { 0.25f, -0.25f },
            new float[] { 0.125f, -0.125f },
            new float[] { 25.0f, -25.0f },
            new float[] { -0.125f, 0.125f },
            new float[] { -0.25f, 0.25f },
            new float[] { -25.0f, 25.0f }
        );
        float[] queryVector = new float[] { -2.0f, 2.0f };
        List<Float> dotProducts = List.of(0.0f, -1.0f, -0.5f, -100.0f, 0.5f, 1.0f, 100.0f);

        for (int i = 0; i < dataVectors.size(); i++) {
            assertEquals(
                KNNEngine.FAISS.score(dotProducts.get(i), SpaceType.INNER_PRODUCT),
                SpaceType.INNER_PRODUCT.getVectorSimilarityFunction().compare(queryVector, dataVectors.get(i)),
                0.0000001
            );
        }
    }
}

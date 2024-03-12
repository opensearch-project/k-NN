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

import java.util.List;

import static org.apache.lucene.util.VectorUtil.scaleMaxInnerProductScore;

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
        List<Float> negativeDotProductScores = List.of(0.0f, -1.0f, -0.5f, -100.0f, 0.5f, 1.0f, 100.0f);
        for (Float negativeDotProduct : negativeDotProductScores) {
            assertEquals(
                SpaceType.INNER_PRODUCT.scoreTranslation(negativeDotProduct),
                scaleMaxInnerProductScore(-1 * negativeDotProduct),
                0.0000001
            );
        }
    }
}

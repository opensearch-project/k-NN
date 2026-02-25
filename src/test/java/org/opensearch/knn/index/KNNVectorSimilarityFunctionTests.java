/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import junit.framework.TestCase;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.knn.plugin.script.KNNScoringUtil;

import java.util.Set;

import static org.apache.lucene.tests.util.LuceneTestCase.expectThrows;
import static org.opensearch.knn.index.KNNVectorSimilarityFunction.COSINE;
import static org.opensearch.knn.index.KNNVectorSimilarityFunction.DOT_PRODUCT;
import static org.opensearch.knn.index.KNNVectorSimilarityFunction.EUCLIDEAN;
import static org.opensearch.knn.index.KNNVectorSimilarityFunction.HAMMING;
import static org.opensearch.knn.index.KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;

public class KNNVectorSimilarityFunctionTests extends TestCase {
    private static final Set<KNNVectorSimilarityFunction> FUNCTION_SET_BACKED_BY_LUCENE = Set.of(
        EUCLIDEAN,
        DOT_PRODUCT,
        COSINE,
        MAXIMUM_INNER_PRODUCT
    );

    public void testFunctions_whenBackedByLucene_thenSameAsLucene() {
        float[] f1 = new float[] { 1.5f, 2.5f, 3.5f, 4.5f, 5.5f };
        float[] f2 = new float[] { 6.5f, 7.5f, 8.5f, 09.5f, 10.5f };
        byte[] b1 = new byte[] { 1, 2, 3 };
        byte[] b2 = new byte[] { 4, 5, 6 };
        for (KNNVectorSimilarityFunction function : KNNVectorSimilarityFunction.values()) {
            if (FUNCTION_SET_BACKED_BY_LUCENE.contains(function) == false) {
                continue;
            }
            assertEquals(VectorSimilarityFunction.valueOf(function.name()), function.getVectorSimilarityFunction());
            assertEquals(function.getVectorSimilarityFunction().compare(f1, f2), function.compare(f1, f2));
            assertEquals(function.getVectorSimilarityFunction().compare(b1, b2), function.compare(b1, b2));
        }
    }

    public void testFunctions_whenHamming_thenFloatVectorIsNotSupported() {
        float[] f1 = new float[] { 1.5f, 2.5f, 3.5f, 4.5f, 5.5f };
        float[] f2 = new float[] { 6.5f, 7.5f, 8.5f, 09.5f, 10.5f };

        Exception ex = expectThrows(IllegalStateException.class, () -> HAMMING.compare(f1, f2));
        assertTrue(ex.getMessage().contains("not supported"));
    }

    public void testFunctions_whenHamming_thenReturnCorrectScore() {
        byte[] b1 = new byte[] { 1, 2, 3 };
        byte[] b2 = new byte[] { 4, 5, 6 };
        assertEquals(1.0f / (1 + KNNScoringUtil.calculateHammingBit(b1, b2)), HAMMING.compare(b1, b2));
    }
}

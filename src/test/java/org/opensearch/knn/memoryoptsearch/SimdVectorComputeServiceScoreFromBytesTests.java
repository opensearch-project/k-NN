/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.junit.Test;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.jni.SimdVectorComputeService;

import java.util.concurrent.ThreadLocalRandom;

public class SimdVectorComputeServiceScoreFromBytesTests extends KNNTestCase {
    @Test
    public void maxIP_singleVector() {
        doScoreFromBytesTest(
            SimdVectorComputeService.SimilarityFunctionType.FP16_MAXIMUM_INNER_PRODUCT,
            KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
            1
        );
    }

    @Test
    public void l2_singleVector() {
        doScoreFromBytesTest(SimdVectorComputeService.SimilarityFunctionType.FP16_L2, KNNVectorSimilarityFunction.EUCLIDEAN, 1);
    }

    @Test
    public void maxIP_bulk() {
        doScoreFromBytesTest(
            SimdVectorComputeService.SimilarityFunctionType.FP16_MAXIMUM_INNER_PRODUCT,
            KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
            37
        );
    }

    @Test
    public void l2_bulk() {
        doScoreFromBytesTest(SimdVectorComputeService.SimilarityFunctionType.FP16_L2, KNNVectorSimilarityFunction.EUCLIDEAN, 37);
    }

    @SneakyThrows
    private void doScoreFromBytesTest(
        final SimdVectorComputeService.SimilarityFunctionType functionType,
        final KNNVectorSimilarityFunction similarityFunction,
        final int numVectors
    ) {
        final int dimension = 123;

        float[] query = new float[dimension];
        for (int j = 0; j < dimension; ++j) {
            query[j] = Float.float16ToFloat(Float.floatToFloat16(ThreadLocalRandom.current().nextFloat()));
        }

        float[][] vectors = new float[numVectors][dimension];
        byte[] fp16Vectors = new byte[numVectors * dimension * Short.BYTES];
        for (int i = 0; i < numVectors; ++i) {
            for (int j = 0; j < dimension; ++j) {
                final short fp16Value = Float.floatToFloat16(ThreadLocalRandom.current().nextFloat());
                vectors[i][j] = Float.float16ToFloat(fp16Value);
                final int byteOffset = i * dimension * Short.BYTES + 2 * j;
                fp16Vectors[byteOffset] = (byte) (fp16Value & 0xFF);
                fp16Vectors[byteOffset + 1] = (byte) ((fp16Value >> 8) & 0xFF);
            }
        }

        float[] scores = new float[numVectors];
        final float maxScore = SimdVectorComputeService.scoreSimilarityInBulkFromBytes(
            query,
            fp16Vectors,
            dimension,
            functionType.ordinal(),
            numVectors,
            scores
        );

        float expectedMaxScore = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < numVectors; ++i) {
            final float expectedScore = similarityFunction.compare(query, vectors[i]);
            assertEquals("Vector " + i, expectedScore, scores[i], 1e-3);
            expectedMaxScore = Math.max(expectedMaxScore, expectedScore);
        }
        assertEquals(expectedMaxScore, maxScore, 1e-3);
    }
}

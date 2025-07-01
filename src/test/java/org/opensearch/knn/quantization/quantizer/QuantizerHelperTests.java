/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.MultiBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;
import oshi.util.tuples.Pair;

import java.io.IOException;

import static org.opensearch.knn.quantization.quantizer.QuantizerHelper.calculateMeanAndStdDev;
import static org.opensearch.knn.quantization.quantizer.QuantizerHelper.calculateThresholds;

public class QuantizerHelperTests extends KNNTestCase {

    public void testCalculateMeanAndStdDev() throws IOException {
        float[][] vectors = { { 1f, 2f }, { 3f, 4f }, { 5f, 6f } };
        ScalarQuantizationParams params = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();
        TrainingRequest<float[]> request = new MockTrainingRequest(params, vectors);
        int[] sampledIndices = { 0, 1, 2 };

        Pair<float[], float[]> result = calculateMeanAndStdDev(request, sampledIndices);

        assertArrayEquals(new float[] { 3f, 4f }, result.getA(), 0.01f);
        assertArrayEquals(new float[] { (float) Math.sqrt(8f / 3), (float) Math.sqrt(8f / 3) }, result.getB(), 0.01f);
    }

    public void testCalculateMeanAndStdDevWithRotation() throws IOException {
        float[][] vectors = { { 1f, 2f }, { 3f, 4f }, { 5f, 6f } };
        ScalarQuantizationParams params = ScalarQuantizationParams.builder()
            .sqType(ScalarQuantizationType.ONE_BIT)
            .enableRandomRotation(true)
            .build();
        TrainingRequest<float[]> request = new MockTrainingRequest(params, vectors);
        int[] sampledIndices = { 0, 1, 2 };
        float[][] rotationMatrix = { { 0f, 1f }, { -1f, 0f } };

        Pair<float[], float[]> result = calculateMeanAndStdDev(request, sampledIndices, rotationMatrix);

        assertArrayEquals(new float[] { 4.0f, -3f }, result.getA(), 0.01f);
        assertArrayEquals(new float[] { (float) Math.sqrt(8f / 3), (float) Math.sqrt(8f / 3) }, result.getB(), 0.01f);

        // try with a more complicated rotation matrix
        float entry = (float) (1f / Math.sqrt(2f));
        float[][] rotationMatrix2 = { { entry, -entry }, { entry, entry } };

        Pair<float[], float[]> result2 = calculateMeanAndStdDev(request, sampledIndices, rotationMatrix2);

        assertArrayEquals(new float[] { -0.707f, 4.949f }, result2.getA(), 0.01f);
        assertArrayEquals(new float[] { 0f, 2.31f }, result2.getB(), 0.01f);
    }

    public void testCalculateOneBitQuantizationState_basicFlow() throws IOException {
        float[][] vectors = { { 1f, 2f }, { 2f, 4f }, { 3f, 6f } };
        ScalarQuantizationParams params = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();
        TrainingRequest<float[]> request = new MockTrainingRequest(params, vectors);
        int[] sampledIndices = { 0, 1, 2 };

        OneBitScalarQuantizationState state = QuantizerHelper.calculateQuantizationState(request, sampledIndices, params);

        assertNotNull(state.getMeanThresholds());
        assertNotNull(state.getAboveThresholdMeans());
        assertNotNull(state.getBelowThresholdMeans());
    }

    public void testCalculateMultiBitQuantizationState_basicFlow() throws IOException {
        float[][] vectors = { { 1f, 2f }, { 2f, 4f }, { 3f, 6f } };
        ScalarQuantizationParams params = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.TWO_BIT).build();
        TrainingRequest<float[]> request = new MockTrainingRequest(params, vectors);
        int[] sampledIndices = { 0, 1, 2 };

        MultiBitScalarQuantizationState state = QuantizerHelper.calculateQuantizationState(request, sampledIndices, params, 2);

        assertNotNull(state.getThresholds());
    }

    public void testThresholdGenerationForMultiBitQuantizer() {
        float[] mean = { 1f, 2f };
        float[] std = { 0.5f, 0.5f };
        float[][] thresholds = invokeThresholds(mean, std, 2);

        assertEquals(2, thresholds.length);
        assertEquals(0.8333334, thresholds[0][0], 0.1f); // near lower bound
        assertEquals(1.8333334, thresholds[0][1], 0.1f);
    }

    public void testRotationMatrixApplication() {
        float[] vector = { 1f, 0f, 0f };
        float[][] rotationMatrix = RandomGaussianRotation.generateRotationMatrix(3);
        float[] rotated = RandomGaussianRotation.applyRotation(vector, rotationMatrix);

        assertEquals(3, rotated.length);
    }

    public void testAboveAndBelowThresholdMeans() throws IOException {
        float[][] vectors = { { 1f, 2f }, { 3f, 4f }, { 5f, 6f } };
        ScalarQuantizationParams params = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();
        TrainingRequest<float[]> request = new MockTrainingRequest(params, vectors);
        int[] sampledIndices = { 0, 1, 2 };

        OneBitScalarQuantizationState state = QuantizerHelper.calculateQuantizationState(request, sampledIndices, params);

        float[] meanThresholds = state.getMeanThresholds();
        float[] aboveThresholdMeans = state.getAboveThresholdMeans();
        float[] belowThresholdMeans = state.getBelowThresholdMeans();

        assertArrayEquals(new float[] { 3f, 4f }, meanThresholds, 0.01f);
        assertArrayEquals(new float[] { 5f, 6f }, aboveThresholdMeans, 0.01f);
        assertArrayEquals(new float[] { 2f, 3f }, belowThresholdMeans, 0.01f);
    }

    public void testAboveAndBelowThresholdMeansWithRotation() throws IOException {
        float[][] vectors = { { 1f, 2f }, { 3f, 4f }, { 5f, 6f } };
        ScalarQuantizationParams params = ScalarQuantizationParams.builder()
            .sqType(ScalarQuantizationType.ONE_BIT)
            .enableRandomRotation(true)
            .build();
        float[][] rotationMatrix = { { 0f, 1f }, { -1f, 0f } };
        TrainingRequest<float[]> request = new MockTrainingRequest(params, vectors, rotationMatrix);
        int[] sampledIndices = { 0, 1, 2 };

        Pair<float[], float[]> meanStd = calculateMeanAndStdDev(request, sampledIndices, rotationMatrix);
        float[][] thresholds = calculateThresholds(meanStd.getA(), meanStd.getB(), 1);

        Pair<float[], float[]> belowAboveMeans = QuantizerHelper.calculateBelowAboveThresholdMeans(
            request,
            thresholds[0],
            sampledIndices,
            rotationMatrix
        );

        assertNotNull(belowAboveMeans.getA());
        assertNotNull(belowAboveMeans.getB());
        assertArrayEquals(new float[] { 4.0f, -3f }, thresholds[0], 0.01f);
        assertArrayEquals(new float[] { 3f, -4f }, belowAboveMeans.getA(), 0.01f);
        assertArrayEquals(new float[] { 6f, -1f }, belowAboveMeans.getB(), 0.01f);
    }

    public void testThrowsOnEmptySampleIndices() {
        ScalarQuantizationParams params = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();
        TrainingRequest<float[]> request = new MockTrainingRequest(params, new float[][] {});
        int[] empty = new int[0];

        expectThrows(IllegalArgumentException.class, () -> QuantizerHelper.calculateQuantizationState(request, empty, params));
    }

    private float[][] invokeThresholds(float[] mean, float[] stdDev, int bitsPerCoordinate) {
        try {
            var method = QuantizerHelper.class.getDeclaredMethod("calculateThresholds", float[].class, float[].class, int.class);
            method.setAccessible(true); // <- This is the key fix
            return (float[][]) method.invoke(null, mean, stdDev, bitsPerCoordinate);
        } catch (Exception e) {
            fail("Failed to invoke calculateThresholds: " + e.getMessage());
            return null;
        }
    }

    private static class MockTrainingRequest extends TrainingRequest<float[]> {
        private final float[][] vectors;
        private final float[][] rotationMatrix;

        public MockTrainingRequest(ScalarQuantizationParams params, float[][] vectors) {
            super(vectors.length, params.getSqType() == ScalarQuantizationType.ONE_BIT && params.isEnableRandomRotation());
            this.vectors = vectors;
            this.rotationMatrix = null;
        }

        public MockTrainingRequest(ScalarQuantizationParams params, float[][] vectors, float[][] rotationMatrix) {
            super(vectors.length, params.getSqType() == ScalarQuantizationType.ONE_BIT && params.isEnableRandomRotation());
            this.vectors = vectors;
            this.rotationMatrix = rotationMatrix;
        }

        @Override
        public float[] getVectorAtThePosition(int position) {
            return vectors[position];
        }

        @Override
        public void resetVectorValues() {
            // No-op for mock
        }

        private float[] applyRotation(float[] vector, float[][] rotationMatrix) {
            float[] result = new float[vector.length];
            for (int i = 0; i < rotationMatrix.length; i++) {
                for (int j = 0; j < vector.length; j++) {
                    result[i] += rotationMatrix[i][j] * vector[j];
                }
            }
            return result;
        }
    }
}

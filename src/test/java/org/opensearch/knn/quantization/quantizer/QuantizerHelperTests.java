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

public class QuantizerHelperTests extends KNNTestCase {

    public void testCalculateMeanAndStdDev() throws IOException {
        float[][] vectors = { { 1f, 2f }, { 3f, 4f }, { 5f, 6f } };
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        TrainingRequest<float[]> request = new MockTrainingRequest(params, vectors);
        int[] sampledIndices = { 0, 1, 2 };

        Pair<float[], float[]> result = QuantizerHelper.calculateMeanAndStdDev(request, sampledIndices);

        assertArrayEquals(new float[] { 3f, 4f }, result.getA(), 0.01f);
        assertArrayEquals(new float[] { (float) Math.sqrt(8f / 3), (float) Math.sqrt(8f / 3) }, result.getB(), 0.01f);
    }

    public void testCalculateMeanAndL2L1Ratio() throws IOException {
        float[][] vectors = { { 1f, 2f }, { 3f, 4f }, { 5f, 6f } };
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        TrainingRequest<float[]> request = new MockTrainingRequest(params, vectors);
        int[] sampledIndices = { 0, 1, 2 };

        Pair<float[], Double> result = QuantizerHelper.calculateMeanAndL2L1Ratio(request, sampledIndices);

        assertArrayEquals(new float[] { 3f, 4f }, result.getA(), 0.01f);
        assertTrue(result.getB() > 0.0);
    }

    public void testCalculateOneBitQuantizationState_basicFlow() throws IOException {
        float[][] vectors = { { 1f, 2f }, { 2f, 4f }, { 3f, 6f } };
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        TrainingRequest<float[]> request = new MockTrainingRequest(params, vectors);
        int[] sampledIndices = { 0, 1, 2 };

        OneBitScalarQuantizationState state = QuantizerHelper.calculateQuantizationState(request, sampledIndices, params);

        assertNotNull(state.getMeanThresholds());
        assertNotNull(state.getAboveThresholdMeans());
        assertNotNull(state.getBelowThresholdMeans());
    }

    public void testCalculateMultiBitQuantizationState_basicFlow() throws IOException {
        float[][] vectors = { { 1f, 2f }, { 2f, 4f }, { 3f, 6f } };
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT);
        TrainingRequest<float[]> request = new MockTrainingRequest(params, vectors);
        int[] sampledIndices = { 0, 1, 2 };

        MultiBitScalarQuantizationState state = QuantizerHelper.calculateQuantizationState(request, sampledIndices, params, 2);

        assertNotNull(state.getThresholds());
        assertNotNull(state.getAboveThresholdMeans());
        assertNotNull(state.getBelowThresholdMeans());
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

    public void testThrowsOnEmptySampleIndices() {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
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

        public MockTrainingRequest(ScalarQuantizationParams params, float[][] vectors) {
            super(vectors.length);
            this.vectors = vectors;
        }

        @Override
        public float[] getVectorAtThePosition(int position) {
            return vectors[position];
        }

        @Override
        public void resetVectorValues() {
            // No-op for mock
        }
    }
}

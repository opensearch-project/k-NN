/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationOutput.BinaryQuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.MultiBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;

import java.io.IOException;
import java.util.Arrays;

public class MultiBitScalarQuantizerTests extends KNNTestCase {

    public void testTrain_twoBit() throws IOException {
        float[][] vectors = {
            { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f },
            { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f },
            { 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f } };

        MultiBitScalarQuantizer quantizer = new MultiBitScalarQuantizer(2);
        TrainingRequest<float[]> request = new MockTrainingRequest(new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT), vectors);
        MultiBitScalarQuantizationState state = (MultiBitScalarQuantizationState) quantizer.train(request);

        assertEquals(2, state.getThresholds().length);
        assertNotNull(state.getBelowThresholdMeans());
        assertNotNull(state.getAboveThresholdMeans());
    }

    public void testTrain_fourBit_withRotationMatrix() throws IOException {
        float[][] vectors = new float[1000][8];
        for (int i = 0; i < 1000; i++)
            Arrays.fill(vectors[i], i);

        MultiBitScalarQuantizer quantizer = new MultiBitScalarQuantizer(4);
        TrainingRequest<float[]> request = new MockTrainingRequest(new ScalarQuantizationParams(ScalarQuantizationType.FOUR_BIT), vectors);
        MultiBitScalarQuantizationState state = (MultiBitScalarQuantizationState) quantizer.train(request);

        assertEquals(4, state.getThresholds().length);
        assertNull(state.getRotationMatrix());
    }

    public void testQuantize_twoBit() throws IOException {
        float[] vector = { 1.3f, 2.2f, 3.3f, 4.1f, 5.6f, 6.7f, 7.4f, 8.1f };
        float[][] thresholds = { { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f }, { 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f } };

        MultiBitScalarQuantizer quantizer = new MultiBitScalarQuantizer(2);
        BinaryQuantizationOutput output = new BinaryQuantizationOutput(2);

        MultiBitScalarQuantizationState state = MultiBitScalarQuantizationState.builder()
            .quantizationParams(new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT))
            .thresholds(thresholds)
            .build();

        quantizer.quantize(vector, state, output);
        assertNotNull(output.getQuantizedVector());
    }

    public void testQuantize_fourBit() throws IOException {
        float[] vector = { 1.3f, 2.2f, 3.3f, 4.1f, 5.6f, 6.7f, 7.4f, 8.1f };
        float[][] thresholds = {
            { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f },
            { 1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f, 7.1f, 8.1f },
            { 1.2f, 2.2f, 3.2f, 4.2f, 5.2f, 6.2f, 7.2f, 8.2f },
            { 1.3f, 2.3f, 3.3f, 4.3f, 5.3f, 6.3f, 7.3f, 8.3f } };

        MultiBitScalarQuantizer quantizer = new MultiBitScalarQuantizer(4);
        BinaryQuantizationOutput output = new BinaryQuantizationOutput(4);

        MultiBitScalarQuantizationState state = MultiBitScalarQuantizationState.builder()
            .quantizationParams(new ScalarQuantizationParams(ScalarQuantizationType.FOUR_BIT))
            .thresholds(thresholds)
            .build();

        quantizer.quantize(vector, state, output);
        assertNotNull(output.getQuantizedVector());
    }

    public void testQuantize_withNullVector_throws() {
        MultiBitScalarQuantizer quantizer = new MultiBitScalarQuantizer(2);
        BinaryQuantizationOutput output = new BinaryQuantizationOutput(2);

        QuantizationState state = MultiBitScalarQuantizationState.builder()
            .quantizationParams(new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT))
            .thresholds(new float[2][8])
            .build();

        expectThrows(IllegalArgumentException.class, () -> quantizer.quantize(null, state, output));
    }

    public void testQuantize_withThresholdDimensionMismatch() {
        MultiBitScalarQuantizer quantizer = new MultiBitScalarQuantizer(2);
        float[] vector = new float[10];
        float[][] thresholds = new float[2][8];

        MultiBitScalarQuantizationState state = MultiBitScalarQuantizationState.builder()
            .quantizationParams(new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT))
            .thresholds(thresholds)
            .build();

        BinaryQuantizationOutput output = new BinaryQuantizationOutput(2);
        expectThrows(IllegalArgumentException.class, () -> quantizer.quantize(vector, state, output));
    }

    public void testQuantize_twoBit_multipleTimes_idempotent() throws IOException {
        MultiBitScalarQuantizer quantizer = new MultiBitScalarQuantizer(2);
        float[] vector = { -2.5f, 1.5f, -0.5f, 4.0f, 6.5f, -3.5f, 0.0f, 7.2f };
        float[][] thresholds = {
            { -3.0f, 1.0f, -1.0f, 3.5f, 5.0f, -4.0f, 0.5f, 7.0f },
            { -2.0f, 2.0f, 0.0f, 4.5f, 6.0f, -2.5f, -0.5f, 8.0f } };

        MultiBitScalarQuantizationState state = MultiBitScalarQuantizationState.builder()
            .quantizationParams(new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT))
            .thresholds(thresholds)
            .build();

        BinaryQuantizationOutput output = new BinaryQuantizationOutput(2);
        quantizer.quantize(vector, state, output);
        byte[] first = output.getQuantizedVector();

        quantizer.quantize(vector, state, output);
        byte[] second = output.getQuantizedVector();

        assertSame(first, second);
    }

    public void testTrain_shouldComputeBelowAboveMeansCorrectly() throws IOException {
        float[][] vectors = { { 1f, 2f, 3f, 4f }, { 2f, 3f, 4f, 5f }, { 3f, 4f, 5f, 6f }, { 9f, 9f, 9f, 9f } };

        MultiBitScalarQuantizer quantizer = new MultiBitScalarQuantizer(2);
        TrainingRequest<float[]> request = new MockTrainingRequest(new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT), vectors);
        MultiBitScalarQuantizationState state = (MultiBitScalarQuantizationState) quantizer.train(request);

        for (float f : state.getAboveThresholdMeans())
            assertTrue(f > 5.0f);
        for (float f : state.getBelowThresholdMeans())
            assertTrue(f < 5.0f);
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
            // No-op
        }
    }
}

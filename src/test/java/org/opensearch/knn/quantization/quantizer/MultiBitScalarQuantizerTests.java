/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationOutput.QuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationParams.SQParams;
import org.opensearch.knn.quantization.models.quantizationState.MultiBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;

public class MultiBitScalarQuantizerTests extends KNNTestCase {

    public void testTrain_twoBit() {
        float[][] vectors = {
            { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f },
            { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f },
            { 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f } };
        MultiBitScalarQuantizer twoBitQuantizer = new MultiBitScalarQuantizer(2);
        int[] sampledIndices = { 0, 1, 2 };
        SQParams params = new SQParams(ScalarQuantizationType.TWO_BIT);
        TrainingRequest<float[]> request = new MockTrainingRequest(params, vectors);
        request.setSampledIndices(sampledIndices);
        QuantizationState state = twoBitQuantizer.train(request);

        assertTrue(state instanceof MultiBitScalarQuantizationState);
        MultiBitScalarQuantizationState mbState = (MultiBitScalarQuantizationState) state;
        assertNotNull(mbState.getThresholds());
        assertEquals(2, mbState.getThresholds().length); // 2-bit quantization should have 2 thresholds
    }

    public void testTrain_fourBit() {
        MultiBitScalarQuantizer fourBitQuantizer = new MultiBitScalarQuantizer(4);
        float[][] vectors = {
            { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f },
            { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f },
            { 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f } };
        int[] sampledIndices = { 0, 1, 2 };
        SQParams params = new SQParams(ScalarQuantizationType.FOUR_BIT);
        TrainingRequest<float[]> request = new MockTrainingRequest(params, vectors);
        request.setSampledIndices(sampledIndices);
        QuantizationState state = fourBitQuantizer.train(request);

        assertTrue(state instanceof MultiBitScalarQuantizationState);
        MultiBitScalarQuantizationState mbState = (MultiBitScalarQuantizationState) state;
        assertNotNull(mbState.getThresholds());
        assertEquals(4, mbState.getThresholds().length); // 4-bit quantization should have 4 thresholds
    }

    public void testQuantize_twoBit() {
        MultiBitScalarQuantizer twoBitQuantizer = new MultiBitScalarQuantizer(2);
        float[] vector = { 1.3f, 2.2f, 3.3f, 4.1f, 5.6f, 6.7f, 7.4f, 8.1f };
        float[][] thresholds = { { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f }, { 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f } };
        SQParams params = new SQParams(ScalarQuantizationType.TWO_BIT);
        MultiBitScalarQuantizationState state = new MultiBitScalarQuantizationState(params, thresholds);

        QuantizationOutput<byte[]> output = twoBitQuantizer.quantize(vector, state);
        assertNotNull(output.getQuantizedVector());
        assertEquals(2, output.getQuantizedVector().length);
    }

    public void testQuantize_fourBit() {
        MultiBitScalarQuantizer fourBitQuantizer = new MultiBitScalarQuantizer(4);
        float[] vector = { 1.3f, 2.2f, 3.3f, 4.1f, 5.6f, 6.7f, 7.4f, 8.1f };
        float[][] thresholds = {
            { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f },
            { 1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f, 7.1f, 8.1f },
            { 1.2f, 2.2f, 3.2f, 4.2f, 5.2f, 6.2f, 7.2f, 8.2f },
            { 1.3f, 2.3f, 3.3f, 4.3f, 5.3f, 6.3f, 7.3f, 8.3f } };
        SQParams params = new SQParams(ScalarQuantizationType.FOUR_BIT);
        MultiBitScalarQuantizationState state = new MultiBitScalarQuantizationState(params, thresholds);

        QuantizationOutput<byte[]> output = fourBitQuantizer.quantize(vector, state);
        assertEquals(4, output.getQuantizedVector().length);
        assertNotNull(output.getQuantizedVector());
    }

    public void testQuantize_withNullVector() {
        MultiBitScalarQuantizer twoBitQuantizer = new MultiBitScalarQuantizer(2);
        expectThrows(
            IllegalArgumentException.class,
            () -> twoBitQuantizer.quantize(
                null,
                new MultiBitScalarQuantizationState(new SQParams(ScalarQuantizationType.TWO_BIT), new float[2][8])
            )
        );
    }

    public void testQuantize_withInvalidState() {
        MultiBitScalarQuantizer twoBitQuantizer = new MultiBitScalarQuantizer(2);
        float[] vector = { 1.3f, 2.2f, 3.3f, 4.1f, 5.6f, 6.7f, 7.4f, 8.1f };
        QuantizationState invalidState = new MockInvalidQuantizationState();
        expectThrows(IllegalArgumentException.class, () -> twoBitQuantizer.quantize(vector, invalidState));
    }

    // Mock classes for testing
    private static class MockTrainingRequest extends TrainingRequest<float[]> {
        private final float[][] vectors;

        public MockTrainingRequest(SQParams params, float[][] vectors) {
            super(params, vectors.length);
            this.vectors = vectors;
        }

        @Override
        public float[] getVectorByDocId(int docId) {
            return vectors[docId];
        }
    }

    private static class MockInvalidQuantizationState implements QuantizationState {
        @Override
        public SQParams getQuantizationParams() {
            return new SQParams(ScalarQuantizationType.UNSUPPORTED_TYPE);
        }

        @Override
        public byte[] toByteArray() {
            return new byte[0];
        }
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationOutput.BinaryQuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationOutput.QuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;
import org.opensearch.knn.quantization.sampler.Sampler;
import org.opensearch.knn.quantization.sampler.SamplerType;
import org.opensearch.knn.quantization.sampler.SamplingFactory;

import java.io.IOException;

public class OneBitScalarQuantizerTests extends KNNTestCase {

    public void testTrain_withTrainingRequired() throws IOException {
        float[][] vectors = { { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f }, { 7.0f, 8.0f, 9.0f } };

        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        TrainingRequest<float[]> originalRequest = new TrainingRequest<float[]>(vectors.length) {
            @Override
            public float[] getVectorAtThePosition(int position) {
                return vectors[position];
            }

            @Override
            public void resetVectorValues() {
                // No-op
            }
        };
        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        QuantizationState state = quantizer.train(originalRequest);

        assertTrue(state instanceof OneBitScalarQuantizationState);
        float[] meanThresholds = ((OneBitScalarQuantizationState) state).getMeanThresholds();
        assertArrayEquals(new float[] { 4.0f, 5.0f, 6.0f }, meanThresholds, 0.001f);
    }

    public void testTrain_withBelowAboveThresholdMeans() throws IOException {
        float[][] vectors = { { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f }, { 7.0f, 8.0f, 9.0f } };
        TrainingRequest<float[]> trainingRequest = new TrainingRequest<>(vectors.length) {
            @Override
            public float[] getVectorAtThePosition(int position) {
                return vectors[position];
            }

            @Override
            public void resetVectorValues() {
                // No-op
            }
        };
        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        QuantizationState state = quantizer.train(trainingRequest);

        assertTrue(state instanceof OneBitScalarQuantizationState);
        OneBitScalarQuantizationState oneBitState = (OneBitScalarQuantizationState) state;

        float[] expectedMeanThresholds = { 4.0f, 5.0f, 6.0f };
        assertArrayEquals(expectedMeanThresholds, oneBitState.getMeanThresholds(), 0.001f);

        // Validate below and above thresholds
        float[] expectedBelowThresholdMeans = { 2.5f, 3.5f, 4.5f };
        float[] expectedAboveThresholdMeans = { 7.0f, 8.0f, 9.0f };
        assertArrayEquals(expectedBelowThresholdMeans, oneBitState.getBelowThresholdMeans(), 0.001f);
        assertArrayEquals(expectedAboveThresholdMeans, oneBitState.getAboveThresholdMeans(), 0.001f);
    }

    public void testQuantize_withState() throws IOException {
        float[] vector = { 3.0f, 6.0f, 9.0f };
        float[] thresholds = { 4.0f, 5.0f, 6.0f };
        OneBitScalarQuantizationState state = OneBitScalarQuantizationState.builder()
            .quantizationParams(new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT))
            .meanThresholds(thresholds)
            .build();

        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        BinaryQuantizationOutput output = new BinaryQuantizationOutput(1);

        quantizer.quantize(vector, state, output);

        assertNotNull(output);
        byte[] expectedPackedBits = new byte[] { 0b01100000 };  // or 96 in decimal
        assertArrayEquals(expectedPackedBits, output.getQuantizedVector());
    }

    public void testQuantize_withNullVector() throws IOException {
        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        OneBitScalarQuantizationState state = OneBitScalarQuantizationState.builder()
            .quantizationParams(new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT))
            .meanThresholds(new float[] { 0.0f })
            .build();
        BinaryQuantizationOutput output = new BinaryQuantizationOutput(1);
        expectThrows(IllegalArgumentException.class, () -> quantizer.quantize(null, state, output));
    }

    public void testQuantize_withInvalidState() throws IOException {
        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        float[] vector = { 1.0f, 2.0f, 3.0f };
        QuantizationState invalidState = new QuantizationState() {
            @Override
            public ScalarQuantizationParams getQuantizationParams() {
                return new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
            }

            @Override
            public int getBytesPerVector() {
                return 0;
            }

            @Override
            public int getDimensions() {
                return 0;
            }

            @Override
            public long ramBytesUsed() {
                return 0;
            }

            @Override
            public byte[] toByteArray() {
                return new byte[0];
            }

            @Override
            public void writeTo(StreamOutput out) throws IOException {
                // Empty implementation for test
            }
        };
        BinaryQuantizationOutput output = new BinaryQuantizationOutput(1);
        expectThrows(IllegalArgumentException.class, () -> quantizer.quantize(vector, invalidState, output));
    }

    public void testQuantize_withMismatchedDimensions() throws IOException {
        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        float[] vector = { 1.0f, 2.0f, 3.0f };
        float[] thresholds = { 4.0f, 5.0f };
        OneBitScalarQuantizationState state = OneBitScalarQuantizationState.builder()
            .quantizationParams(new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT))
            .meanThresholds(thresholds)
            .build();
        QuantizationOutput<byte[]> output = new BinaryQuantizationOutput(1);
        expectThrows(IllegalArgumentException.class, () -> quantizer.quantize(vector, state, output));
    }

    public void testTrain_withRotationApplied() throws IOException {
        float[][] vectors = { { 10.0f, 200.0f, 3000.0f }, { 4000.0f, 5000.0f, 6000.0f }, { 7000.0f, 8000.0f, 9000.0f } };

        TrainingRequest<float[]> trainingRequest = new TrainingRequest<>(vectors.length) {
            @Override
            public float[] getVectorAtThePosition(int position) {
                return vectors[position];
            }

            @Override
            public void resetVectorValues() {
                // No-op
            }
        };

        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        OneBitScalarQuantizationState state = (OneBitScalarQuantizationState) quantizer.train(trainingRequest);

        assertNotNull(state);
        assertNotNull(state.getRotationMatrix());
        assertTrue(state.getRotationMatrix().length > 0);
    }

    public void testTrain_withoutRotationMatrix() throws IOException {
        float[][] vectors = { { 1.0f, 1.0f, 1.0f }, { 1.1f, 1.1f, 1.1f }, { 0.9f, 0.9f, 0.9f } };

        TrainingRequest<float[]> trainingRequest = new TrainingRequest<>(vectors.length) {
            @Override
            public float[] getVectorAtThePosition(int position) {
                return vectors[position];
            }

            @Override
            public void resetVectorValues() {
                // No-op
            }
        };

        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        OneBitScalarQuantizationState state = (OneBitScalarQuantizationState) quantizer.train(trainingRequest);

        assertNotNull(state);
        assertNull(state.getRotationMatrix());
    }

    public void testQuantize_withRotationMatrix() {
        float[] vector = { 3.0f, 6.0f, 9.0f };
        float[] thresholds = { 4.0f, 5.0f, 6.0f };

        // Generate a rotation matrix
        float[][] rotationMatrix = RandomGaussianRotation.generateRotationMatrix(3);

        OneBitScalarQuantizationState state = OneBitScalarQuantizationState.builder()
            .quantizationParams(new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT))
            .meanThresholds(thresholds)
            .rotationMatrix(rotationMatrix)
            .build();

        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        BinaryQuantizationOutput output = new BinaryQuantizationOutput(1);

        quantizer.quantize(vector, state, output);

        assertNotNull(output);
        assertNotNull(output.getQuantizedVector());
    }

    public void testQuantize_withDifferentRotationMatrices() {
        float[] vector = { 3.0f, 6.0f, 9.0f };
        float[] thresholds = { 4.0f, 5.0f, 6.0f };

        // Generate two different rotation matrices
        float[][] rotationMatrix1 = RandomGaussianRotation.generateRotationMatrix(3);
        float[][] rotationMatrix2 = RandomGaussianRotation.generateRotationMatrix(3);

        OneBitScalarQuantizationState state1 = OneBitScalarQuantizationState.builder()
            .quantizationParams(new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT))
            .meanThresholds(thresholds)
            .rotationMatrix(rotationMatrix1)
            .build();

        OneBitScalarQuantizationState state2 = OneBitScalarQuantizationState.builder()
            .quantizationParams(new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT))
            .meanThresholds(thresholds)
            .rotationMatrix(rotationMatrix2)
            .build();

        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        BinaryQuantizationOutput output1 = new BinaryQuantizationOutput(1);
        BinaryQuantizationOutput output2 = new BinaryQuantizationOutput(1);

        quantizer.quantize(vector, state1, output1);
        quantizer.quantize(vector, state2, output2);

        assertNotNull(output1.getQuantizedVector());
        assertNotNull(output2.getQuantizedVector());
        assertFalse(output1.getQuantizedVector().equals(output2.getQuantizedVector()));
    }

    public void testRotationConsistency() {
        float[] vector = { 5.0f, 10.0f, 15.0f };
        float[] thresholds = { 6.0f, 11.0f, 16.0f };

        // Generate a fixed rotation matrix
        float[][] rotationMatrix = RandomGaussianRotation.generateRotationMatrix(3);

        OneBitScalarQuantizationState state = OneBitScalarQuantizationState.builder()
            .quantizationParams(new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT))
            .meanThresholds(thresholds)
            .rotationMatrix(rotationMatrix)
            .build();

        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        BinaryQuantizationOutput output1 = new BinaryQuantizationOutput(1);
        BinaryQuantizationOutput output2 = new BinaryQuantizationOutput(1);

        quantizer.quantize(vector, state, output1);
        quantizer.quantize(vector, state, output2);

        assertArrayEquals(output1.getQuantizedVector(), output2.getQuantizedVector());
    }

    public void testCalculateMean() throws IOException {
        float[][] vectors = { { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f }, { 7.0f, 8.0f, 9.0f } };

        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        TrainingRequest<float[]> samplingRequest = new TrainingRequest<float[]>(vectors.length) {
            @Override
            public float[] getVectorAtThePosition(int position) {
                return vectors[position];
            }

            @Override
            public void resetVectorValues() {
                // No-op
            }
        };

        Sampler sampler = SamplingFactory.getSampler(SamplerType.RESERVOIR);
        int[] sampledIndices = sampler.sample(vectors.length, 3);
        OneBitScalarQuantizationState oneBitScalarQuantizationState = QuantizerHelper.calculateQuantizationState(
            samplingRequest,
            sampledIndices,
            new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT)
        );
        assertArrayEquals(new float[] { 4.0f, 5.0f, 6.0f }, oneBitScalarQuantizationState.getMeanThresholds(), 0.001f);
    }

    public void testCalculateMean_withNullVector() {
        float[][] vectors = { { 1.0f, 2.0f, 3.0f }, null, { 7.0f, 8.0f, 9.0f } };

        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        TrainingRequest<float[]> samplingRequest = new TrainingRequest<float[]>(vectors.length) {
            @Override
            public float[] getVectorAtThePosition(int position) {
                return vectors[position];
            }

            @Override
            public void resetVectorValues() {
                // No-op
            }
        };

        Sampler sampler = SamplingFactory.getSampler(SamplerType.RESERVOIR);
        int[] sampledIndices = sampler.sample(vectors.length, 3);
        ScalarQuantizationParams quantizationParams = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        expectThrows(
            IllegalArgumentException.class,
            () -> QuantizerHelper.calculateQuantizationState(samplingRequest, sampledIndices, quantizationParams)
        );
    }

    public void testQuantize_withState_multiple_times() throws IOException {
        float[] vector = { 3.0f, 6.0f, 9.0f };
        float[] thresholds = { 4.0f, 5.0f, 6.0f };
        OneBitScalarQuantizationState state = OneBitScalarQuantizationState.builder()
            .quantizationParams(new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT))
            .meanThresholds(thresholds)
            .build();

        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        BinaryQuantizationOutput output = new BinaryQuantizationOutput(1);

        // First quantization
        quantizer.quantize(vector, state, output);
        assertNotNull(output);
        byte[] expectedPackedBits = new byte[] { 0b01100000 };  // or 96 in decimal
        assertArrayEquals(expectedPackedBits, output.getQuantizedVector());

        // Save the reference to the byte array
        byte[] firstByteArray = output.getQuantizedVector();

        // Modify vector and thresholds for a second quantization call
        vector = new float[] { 7.0f, 8.0f, 9.0f };
        thresholds = new float[] { 6.0f, 7.0f, 8.0f };
        state = OneBitScalarQuantizationState.builder()
            .quantizationParams(new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT))
            .meanThresholds(thresholds)
            .build();

        // Second quantization
        output.prepareQuantizedVector(vector.length);  // Ensure it is prepared for the new vector
        quantizer.quantize(vector, state, output);

        // Assert that the same byte array reference is used
        assertSame(firstByteArray, output.getQuantizedVector());

        // Check the new output
        expectedPackedBits = new byte[] { (byte) 0b11100000 };  // or 224 in decimal
        assertArrayEquals(expectedPackedBits, output.getQuantizedVector());
    }

    public void testQuantize_ReuseByteArray() throws IOException {
        float[] vector = { 3.0f, 6.0f, 9.0f };
        float[] thresholds = { 4.0f, 5.0f, 6.0f };
        OneBitScalarQuantizationState state = OneBitScalarQuantizationState.builder()
            .quantizationParams(new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT))
            .meanThresholds(thresholds)
            .build();

        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        BinaryQuantizationOutput output = new BinaryQuantizationOutput(1);
        output.prepareQuantizedVector(vector.length);

        // First quantization
        quantizer.quantize(vector, state, output);
        byte[] firstByteArray = output.getQuantizedVector();

        // Second quantization with the same vector length
        output.prepareQuantizedVector(vector.length);  // Reuse the prepared output
        byte[] secondByteArray = output.getQuantizedVector();

        // Assert that the same byte array reference is used
        assertSame(firstByteArray, secondByteArray);

        // Third quantization with the same vector length
        output.prepareQuantizedVector(vector.length);  // Reuse the prepared output again
        quantizer.quantize(vector, state, output);
        byte[] thirdByteArray = output.getQuantizedVector();

        // Assert that the same byte array reference is still used
        assertSame(firstByteArray, thirdByteArray);
    }

    public void testQuantize_withMultipleVectors_inLoop() throws IOException {
        OneBitScalarQuantizer oneBitQuantizer = new OneBitScalarQuantizer();
        float[][] vectors = { { 1.0f, 2.0f, 3.0f, 4.0f }, { 2.0f, 3.0f, 4.0f, 5.0f }, { 1.5f, 2.5f, 3.5f, 4.5f } };
        float[] thresholds = { 1.5f, 2.5f, 3.5f, 4.5f };

        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        OneBitScalarQuantizationState state = OneBitScalarQuantizationState.builder()
            .quantizationParams(new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT))
            .meanThresholds(thresholds)
            .build();

        BinaryQuantizationOutput output = new BinaryQuantizationOutput(1);

        byte[] previousByteArray = null;
        for (float[] vector : vectors) {
            // Check if output is already prepared before quantization
            boolean wasPrepared = output.isPrepared(vector.length);

            // Prepare the output for the new vector length
            output.prepareQuantizedVector(vector.length);

            // Ensure that if it was prepared, it stays the same reference
            if (wasPrepared) {
                assertSame(previousByteArray, output.getQuantizedVector());
            }

            // Perform the quantization
            oneBitQuantizer.quantize(vector, state, output);

            // Save the reference to the byte array after quantization
            previousByteArray = output.getQuantizedVector();

            // Check that the output vector is correctly prepared
            assertTrue(output.isPrepared(vector.length));
        }
    }
}

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

        ScalarQuantizationParams params = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();
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
            .quantizationParams(ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build())
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
            .quantizationParams(ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build())
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
                return ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();
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
            .quantizationParams(ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build())
            .meanThresholds(thresholds)
            .build();
        QuantizationOutput<byte[]> output = new BinaryQuantizationOutput(1);
        expectThrows(IllegalArgumentException.class, () -> quantizer.quantize(vector, state, output));
    }

    public void testTrain_withRotationApplied() throws IOException {
        float[][] vectors = { { 10.0f, 200.0f, 3000.0f }, { 4000.0f, 5000.0f, 6000.0f }, { 7000.0f, 8000.0f, 9000.0f } };

        TrainingRequest<float[]> trainingRequest = new TrainingRequest<>(vectors.length, true) {
            @Override
            public float[] getVectorAtThePosition(int position) {
                return vectors[position];
            }

            @Override
            public void resetVectorValues() {
                // No-op
            }
        };

        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer(true);
        OneBitScalarQuantizationState state = (OneBitScalarQuantizationState) quantizer.train(trainingRequest);

        assertNotNull(state);
        assertNotNull(state.getRotationMatrix());
        assertTrue(state.getRotationMatrix().length > 0);
    }

    public void testTrain_withoutRotationMatrix() throws IOException {
        float[][] vectors = { { 1.0f, 1.0f, 1.0f }, { 1.1f, 1.1f, 1.1f }, { 0.9f, 0.9f, 0.9f } };

        TrainingRequest<float[]> trainingRequest = new TrainingRequest<>(vectors.length, false) {
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
            .quantizationParams(ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build())
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
            .quantizationParams(ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build())
            .meanThresholds(thresholds)
            .rotationMatrix(rotationMatrix1)
            .build();

        OneBitScalarQuantizationState state2 = OneBitScalarQuantizationState.builder()
            .quantizationParams(ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build())
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
            .quantizationParams(ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build())
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

        ScalarQuantizationParams params = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();
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
            ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build()
        );
        assertArrayEquals(new float[] { 4.0f, 5.0f, 6.0f }, oneBitScalarQuantizationState.getMeanThresholds(), 0.001f);
    }

    public void testCalculateMean_withNullVector() {
        float[][] vectors = { { 1.0f, 2.0f, 3.0f }, null, { 7.0f, 8.0f, 9.0f } };

        ScalarQuantizationParams params = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();
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
        ScalarQuantizationParams quantizationParams = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();
        expectThrows(
            IllegalArgumentException.class,
            () -> QuantizerHelper.calculateQuantizationState(samplingRequest, sampledIndices, quantizationParams)
        );
    }

    public void testQuantize_withState_multiple_times() throws IOException {
        float[] vector = { 3.0f, 6.0f, 9.0f };
        float[] thresholds = { 4.0f, 5.0f, 6.0f };
        OneBitScalarQuantizationState state = OneBitScalarQuantizationState.builder()
            .quantizationParams(ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build())
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
            .quantizationParams(ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build())
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
            .quantizationParams(ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build())
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

    public void testOneBitConfig_producesOneBitPerDimension() {
        ScalarQuantizationParams params = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();
        assertEquals(1, params.getSqType().getId());

        float[] thresholds8D = new float[8];
        OneBitScalarQuantizationState state8D = OneBitScalarQuantizationState.builder()
            .quantizationParams(params)
            .meanThresholds(thresholds8D)
            .build();
        assertEquals(1, state8D.getBytesPerVector());
        assertEquals(8, state8D.getDimensions());

        float[] thresholds16D = new float[16];
        OneBitScalarQuantizationState state16D = OneBitScalarQuantizationState.builder()
            .quantizationParams(params)
            .meanThresholds(thresholds16D)
            .build();
        assertEquals(2, state16D.getBytesPerVector());
        assertEquals(16, state16D.getDimensions());

        float[] thresholds128D = new float[128];
        OneBitScalarQuantizationState state128D = OneBitScalarQuantizationState.builder()
            .quantizationParams(params)
            .meanThresholds(thresholds128D)
            .build();
        assertEquals(16, state128D.getBytesPerVector());
        assertEquals(128, state128D.getDimensions());
        // FP32 uses 4 bytes per dimension; 1-bit uses 1 bit per dimension → 32x compression
        int fp32Bytes = 128 * Float.BYTES;
        assertEquals(32, fp32Bytes / state128D.getBytesPerVector());
    }

    public void testOneBitConfig_unalignedDimensions_bytesPerVector() {
        ScalarQuantizationParams params = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();

        float[] thresholds3D = new float[3];
        OneBitScalarQuantizationState state3D = OneBitScalarQuantizationState.builder()
            .quantizationParams(params)
            .meanThresholds(thresholds3D)
            .build();
        assertEquals(1, state3D.getBytesPerVector());
        assertEquals(8, state3D.getDimensions());

        float[] thresholds9D = new float[9];
        OneBitScalarQuantizationState state9D = OneBitScalarQuantizationState.builder()
            .quantizationParams(params)
            .meanThresholds(thresholds9D)
            .build();
        assertEquals(2, state9D.getBytesPerVector());
        assertEquals(16, state9D.getDimensions());
    }

    public void testOneBitQuantize_roundTripErrorIsBounded() throws IOException {
        float[][] vectors = {
            { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f },
            { 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f },
            { 4.0f, 4.0f, 4.0f, 4.0f, 5.0f, 5.0f, 5.0f, 5.0f },
            { 1.5f, 3.5f, 5.5f, 7.5f, 2.5f, 4.5f, 6.5f, 8.5f } };

        TrainingRequest<float[]> trainingRequest = new TrainingRequest<float[]>(vectors.length) {
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
        float[] thresholds = state.getMeanThresholds();

        BinaryQuantizationOutput output = new BinaryQuantizationOutput(1);
        for (float[] vector : vectors) {
            quantizer.quantize(vector, state, output);
            byte[] quantized = output.getQuantizedVector();

            // Reconstruct: bit=1 → aboveThresholdMean, bit=0 → belowThresholdMean
            float[] aboveMeans = state.getAboveThresholdMeans();
            float[] belowMeans = state.getBelowThresholdMeans();
            float[] reconstructed = new float[vector.length];
            for (int d = 0; d < vector.length; d++) {
                int byteIndex = d >> 3;
                int bitIndex = 7 - (d & 7);
                boolean bitSet = (quantized[byteIndex] & (1 << bitIndex)) != 0;
                reconstructed[d] = bitSet ? aboveMeans[d] : belowMeans[d];
            }

            for (int d = 0; d < vector.length; d++) {
                float error = Math.abs(vector[d] - reconstructed[d]);
                float spread = Math.abs(aboveMeans[d] - belowMeans[d]);
                assertTrue(
                    "Reconstruction error " + error + " exceeds data spread " + spread + " at dimension " + d,
                    error <= spread + 1e-6f
                );
            }
        }
    }

    public void testOneBitQuantize_consistentBitAssignment() throws IOException {
        float[] thresholds = { 4.0f, 5.0f, 6.0f, 7.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        OneBitScalarQuantizationState state = OneBitScalarQuantizationState.builder()
            .quantizationParams(ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build())
            .meanThresholds(thresholds)
            .build();

        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        BinaryQuantizationOutput output = new BinaryQuantizationOutput(1);

        float[] allAbove = { 5.0f, 6.0f, 7.0f, 8.0f, 4.0f, 5.0f, 6.0f, 7.0f };
        quantizer.quantize(allAbove, state, output);
        byte[] result = output.getQuantizedVector();
        assertEquals("All bits should be 1 → 0xFF", (byte) 0xFF, result[0]);

        float[] allBelow = { 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        output.prepareQuantizedVector(allBelow.length);
        quantizer.quantize(allBelow, state, output);
        result = output.getQuantizedVector();
        assertEquals("All bits should be 0 → 0x00", (byte) 0x00, result[0]);
    }

    public void testTrain_withSingleVector() throws IOException {
        float[][] vectors = { { 5.0f, 10.0f, 15.0f } };
        TrainingRequest<float[]> trainingRequest = new TrainingRequest<float[]>(vectors.length) {
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
        assertNotNull(state.getMeanThresholds());
        assertEquals(3, state.getMeanThresholds().length);
        assertArrayEquals(new float[] { 5.0f, 10.0f, 15.0f }, state.getMeanThresholds(), 0.001f);
    }

    public void testTrain_withLargeNumberOfVectors() throws IOException {
        int numVectors = 1000;
        int dimensions = 64;
        float[][] vectors = new float[numVectors][dimensions];
        for (int i = 0; i < numVectors; i++) {
            for (int d = 0; d < dimensions; d++) {
                vectors[i][d] = (float) (i * dimensions + d) / (numVectors * dimensions);
            }
        }

        TrainingRequest<float[]> trainingRequest = new TrainingRequest<float[]>(vectors.length) {
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
        float[] thresholds = state.getMeanThresholds();
        assertEquals(dimensions, thresholds.length);
        assertEquals(8, state.getBytesPerVector());
    }

    public void testTrain_withEmptyVectorSet_throwsException() {
        float[][] vectors = {};
        TrainingRequest<float[]> trainingRequest = new TrainingRequest<float[]>(vectors.length) {
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
        expectThrows(IllegalArgumentException.class, () -> quantizer.train(trainingRequest));
    }

    public void testQuantize_withMultipleVectors_inLoop() throws IOException {
        OneBitScalarQuantizer oneBitQuantizer = new OneBitScalarQuantizer();
        float[][] vectors = { { 1.0f, 2.0f, 3.0f, 4.0f }, { 2.0f, 3.0f, 4.0f, 5.0f }, { 1.5f, 2.5f, 3.5f, 4.5f } };
        float[] thresholds = { 1.5f, 2.5f, 3.5f, 4.5f };

        ScalarQuantizationParams params = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();
        OneBitScalarQuantizationState state = OneBitScalarQuantizationState.builder()
            .quantizationParams(ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build())
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

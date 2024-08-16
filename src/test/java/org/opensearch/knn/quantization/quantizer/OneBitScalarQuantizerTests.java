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

    public void testTrain_withTrainingRequired() {
        float[][] vectors = { { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f }, { 7.0f, 8.0f, 9.0f } };

        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        TrainingRequest<float[]> originalRequest = new TrainingRequest<float[]>(vectors.length) {
            @Override
            public float[] getVectorByDocId(int docId) {
                return vectors[docId];
            }
        };
        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        QuantizationState state = quantizer.train(originalRequest);

        assertTrue(state instanceof OneBitScalarQuantizationState);
        float[] meanThresholds = ((OneBitScalarQuantizationState) state).getMeanThresholds();
        assertArrayEquals(new float[] { 4.0f, 5.0f, 6.0f }, meanThresholds, 0.001f);
    }

    public void testQuantize_withState() throws IOException {
        float[] vector = { 3.0f, 6.0f, 9.0f };
        float[] thresholds = { 4.0f, 5.0f, 6.0f };
        OneBitScalarQuantizationState state = new OneBitScalarQuantizationState(
            new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT),
            thresholds
        );

        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        BinaryQuantizationOutput output = new BinaryQuantizationOutput(1);

        quantizer.quantize(vector, state, output);

        assertNotNull(output);
        byte[] expectedPackedBits = new byte[] { 0b01100000 };  // or 96 in decimal
        assertArrayEquals(expectedPackedBits, output.getQuantizedVector());
    }

    public void testQuantize_withNullVector() throws IOException {
        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        OneBitScalarQuantizationState state = new OneBitScalarQuantizationState(
            new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT),
            new float[] { 0.0f }
        );
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
        OneBitScalarQuantizationState state = new OneBitScalarQuantizationState(
            new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT),
            thresholds
        );
        QuantizationOutput<byte[]> output = new BinaryQuantizationOutput(1);
        expectThrows(IllegalArgumentException.class, () -> quantizer.quantize(vector, state, output));
    }

    public void testCalculateMean() {
        float[][] vectors = { { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f }, { 7.0f, 8.0f, 9.0f } };

        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        TrainingRequest<float[]> samplingRequest = new TrainingRequest<float[]>(vectors.length) {
            @Override
            public float[] getVectorByDocId(int docId) {
                return vectors[docId];
            }
        };

        Sampler sampler = SamplingFactory.getSampler(SamplerType.RESERVOIR);
        int[] sampledIndices = sampler.sample(vectors.length, 3);
        float[] meanThresholds = QuantizerHelper.calculateMeanThresholds(samplingRequest, sampledIndices);
        assertArrayEquals(new float[] { 4.0f, 5.0f, 6.0f }, meanThresholds, 0.001f);
    }

    public void testCalculateMean_withNullVector() {
        float[][] vectors = { { 1.0f, 2.0f, 3.0f }, null, { 7.0f, 8.0f, 9.0f } };

        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        TrainingRequest<float[]> samplingRequest = new TrainingRequest<float[]>(vectors.length) {
            @Override
            public float[] getVectorByDocId(int docId) {
                return vectors[docId];
            }
        };

        Sampler sampler = SamplingFactory.getSampler(SamplerType.RESERVOIR);
        int[] sampledIndices = sampler.sample(vectors.length, 3);
        expectThrows(IllegalArgumentException.class, () -> QuantizerHelper.calculateMeanThresholds(samplingRequest, sampledIndices));
    }

    public void testQuantize_withState_multiple_times() throws IOException {
        float[] vector = { 3.0f, 6.0f, 9.0f };
        float[] thresholds = { 4.0f, 5.0f, 6.0f };
        OneBitScalarQuantizationState state = new OneBitScalarQuantizationState(
            new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT),
            thresholds
        );

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
        state = new OneBitScalarQuantizationState(new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT), thresholds);

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
        OneBitScalarQuantizationState state = new OneBitScalarQuantizationState(
            new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT),
            thresholds
        );

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
        OneBitScalarQuantizationState state = new OneBitScalarQuantizationState(params, thresholds);

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

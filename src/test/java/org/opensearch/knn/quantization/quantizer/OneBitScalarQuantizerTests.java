/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationOutput.QuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationParams.SQParams;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;
import org.opensearch.knn.quantization.sampler.Sampler;
import org.opensearch.knn.quantization.sampler.SamplingFactory;
import org.opensearch.knn.quantization.util.QuantizerHelper;

public class OneBitScalarQuantizerTests extends KNNTestCase {

    public void testTrain_withTrainingRequired() {
        float[][] vectors = { { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f }, { 7.0f, 8.0f, 9.0f } };

        SQParams params = new SQParams(ScalarQuantizationType.ONE_BIT);
        TrainingRequest<float[]> originalRequest = new TrainingRequest<float[]>(params, vectors.length) {
            @Override
            public float[] getVectorByDocId(int docId) {
                return vectors[docId];
            }
        };
        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        QuantizationState state = quantizer.train(originalRequest);

        assertTrue(state instanceof OneBitScalarQuantizationState);
        float[] mean = ((OneBitScalarQuantizationState) state).getMeanThresholds();
        assertArrayEquals(new float[] { 4.0f, 5.0f, 6.0f }, mean, 0.001f);
    }

    public void testQuantize_withState() {
        float[] vector = { 3.0f, 6.0f, 9.0f };
        float[] thresholds = { 4.0f, 5.0f, 6.0f };
        OneBitScalarQuantizationState state = new OneBitScalarQuantizationState(new SQParams(ScalarQuantizationType.ONE_BIT), thresholds);

        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        QuantizationOutput<byte[]> output = quantizer.quantize(vector, state);

        assertNotNull(output);
        byte[] expectedPackedBits = new byte[] { 0b01100000 };  // or 96 in decimal
        assertArrayEquals(expectedPackedBits, output.getQuantizedVector());
    }

    public void testQuantize_withNullVector() {
        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        OneBitScalarQuantizationState state = new OneBitScalarQuantizationState(
            new SQParams(ScalarQuantizationType.ONE_BIT),
            new float[] { 0.0f }
        );
        expectThrows(IllegalArgumentException.class, () -> quantizer.quantize(null, state));
    }

    public void testQuantize_withInvalidState() {
        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        float[] vector = { 1.0f, 2.0f, 3.0f };
        QuantizationState invalidState = new QuantizationState() {
            @Override
            public SQParams getQuantizationParams() {
                return new SQParams(ScalarQuantizationType.ONE_BIT);
            }

            @Override
            public byte[] toByteArray() {
                return new byte[0];
            }
        };
        expectThrows(IllegalArgumentException.class, () -> quantizer.quantize(vector, invalidState));
    }

    public void testQuantize_withMismatchedDimensions() {
        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        float[] vector = { 1.0f, 2.0f, 3.0f };
        float[] thresholds = { 4.0f, 5.0f };
        OneBitScalarQuantizationState state = new OneBitScalarQuantizationState(new SQParams(ScalarQuantizationType.ONE_BIT), thresholds);

        expectThrows(IllegalArgumentException.class, () -> quantizer.quantize(vector, state));
    }

    public void testCalculateMean() {
        float[][] vectors = { { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f }, { 7.0f, 8.0f, 9.0f } };

        SQParams params = new SQParams(ScalarQuantizationType.ONE_BIT);
        TrainingRequest<float[]> samplingRequest = new TrainingRequest<float[]>(params, vectors.length) {
            @Override
            public float[] getVectorByDocId(int docId) {
                return vectors[docId];
            }
        };

        Sampler sampler = SamplingFactory.getSampler(SamplingFactory.SamplerType.RESERVOIR);
        int[] sampledIndices = sampler.sample(vectors.length, 3);
        float[] mean = QuantizerHelper.calculateMean(samplingRequest, sampledIndices);
        assertArrayEquals(new float[] { 4.0f, 5.0f, 6.0f }, mean, 0.001f);
    }

    public void testCalculateMean_withNullVector() {
        float[][] vectors = { { 1.0f, 2.0f, 3.0f }, null, { 7.0f, 8.0f, 9.0f } };

        SQParams params = new SQParams(ScalarQuantizationType.ONE_BIT);
        TrainingRequest<float[]> samplingRequest = new TrainingRequest<float[]>(params, vectors.length) {
            @Override
            public float[] getVectorByDocId(int docId) {
                return vectors[docId];
            }
        };

        Sampler sampler = SamplingFactory.getSampler(SamplingFactory.SamplerType.RESERVOIR);
        int[] sampledIndices = sampler.sample(vectors.length, 3);
        expectThrows(IllegalArgumentException.class, () -> QuantizerHelper.calculateMean(samplingRequest, sampledIndices));
    }
}

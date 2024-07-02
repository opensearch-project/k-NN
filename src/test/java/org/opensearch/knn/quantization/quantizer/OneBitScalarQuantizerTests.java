/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.quantization.quantizer;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.enums.SQTypes;
import org.opensearch.knn.quantization.models.quantizationOutput.QuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationParams.SQParams;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.requests.SamplingTrainingRequest;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;
import org.opensearch.knn.quantization.sampler.ReservoirSampler;

public class OneBitScalarQuantizerTests extends KNNTestCase {

    public void testTrain() {
        float[][] vectors = {
                {1.0f, 2.0f, 3.0f},
                {4.0f, 5.0f, 6.0f},
                {7.0f, 8.0f, 9.0f}
        };

        SQParams params = new SQParams(SQTypes.ONE_BIT);
        TrainingRequest<float[]> originalRequest = new TrainingRequest<float[]>(params, vectors.length) {
            @Override
            public float[] getVector() {
                return null; // Not used in this test
            }
            @Override
            public float[] getVectorByDocId(int docId) {
                return vectors[docId];
            }
        };
        TrainingRequest<float[]> trainingRequest = new SamplingTrainingRequest<>(
                originalRequest,
                new ReservoirSampler(),
                vectors.length
        );
        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        QuantizationState state = quantizer.train(trainingRequest);

        assertTrue(state instanceof OneBitScalarQuantizationState);
        float[] mean = ((OneBitScalarQuantizationState) state).getMean();
        assertArrayEquals(new float[]{4.0f, 5.0f, 6.0f}, mean, 0.001f);
    }

    public void testQuantize_withState() {
        float[] vector = {3.0f, 6.0f, 9.0f};
        float[] thresholds = {4.0f, 5.0f, 6.0f};
        OneBitScalarQuantizationState state = new OneBitScalarQuantizationState(new SQParams(SQTypes.ONE_BIT), thresholds);

        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        QuantizationOutput<byte[]> output = quantizer.quantize(vector, state);

        assertArrayEquals(new byte[]{96}, output.getQuantizedVector());
    }

    public void testQuantize_withoutState() {
        float[] vector = {-1.0f, 0.5f, 1.5f};

        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        QuantizationOutput<byte[]> output = quantizer.quantize(vector);

        assertArrayEquals(new byte[]{96}, output.getQuantizedVector());
    }

    public void testQuantize_withNullVector() {
        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer();
        expectThrows( IllegalArgumentException.class, ()-> quantizer.quantize(null));
    }
}

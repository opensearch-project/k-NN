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

package org.opensearch.knn.quantization;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.enums.SQTypes;
import org.opensearch.knn.quantization.models.quantizationParams.SQParams;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;
import org.opensearch.knn.quantization.quantizer.Quantizer;


public class QuantizationManagerTests extends KNNTestCase {
    public void testSingletonInstance() {
        QuantizationManager instance1 = QuantizationManager.getInstance();
        QuantizationManager instance2 = QuantizationManager.getInstance();
        assertSame(instance1, instance2);
    }

    public void testTrain() {
        QuantizationManager quantizationManager = QuantizationManager.getInstance();
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
        QuantizationState state = quantizationManager.train(originalRequest);

        assertTrue(state instanceof OneBitScalarQuantizationState);
        float[] mean = ((OneBitScalarQuantizationState) state).getMean();
        assertArrayEquals(new float[]{4.0f, 5.0f, 6.0f}, mean, 0.001f);
    }

    public void testTrainWithFewVectors() {
        QuantizationManager quantizationManager = QuantizationManager.getInstance();
        float[][] vectors = {
                {1.0f, 2.0f, 3.0f},
                {4.0f, 5.0f, 6.0f}
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

        QuantizationState state = quantizationManager.train(originalRequest);

        assertTrue(state instanceof OneBitScalarQuantizationState);
        float[] mean = ((OneBitScalarQuantizationState) state).getMean();
        assertArrayEquals(new float[]{2.5f, 3.5f, 4.5f}, mean, 0.001f);
    }


    public void testGetQuantizer() {
        QuantizationManager quantizationManager = QuantizationManager.getInstance();
        SQParams params = new SQParams(SQTypes.ONE_BIT);

        Quantizer<?, ?> quantizer = quantizationManager.getQuantizer(params);

        assertTrue(quantizer instanceof Quantizer);
    }
}

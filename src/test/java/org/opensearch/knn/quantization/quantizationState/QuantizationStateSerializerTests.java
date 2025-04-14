/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizationState;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.MultiBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;

import java.io.IOException;

public class QuantizationStateSerializerTests extends KNNTestCase {

    public void testSerializeAndDeserializeOneBitScalarQuantizationState() throws IOException {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        float[] mean = new float[] { 0.1f, 0.2f, 0.3f };
        float[] below = new float[] { 0.05f, 0.15f, 0.25f };
        float[] above = new float[] { 0.15f, 0.25f, 0.35f };
        double l2l1Ratio = 0.7;
        float[][] rotationMatrix = new float[][] {
                { 1.0f, 0.0f, 0.0f },
                { 0.0f, 1.0f, 0.0f },
                { 0.0f, 0.0f, 1.0f }
        };

        OneBitScalarQuantizationState state = OneBitScalarQuantizationState.builder()
                .quantizationParams(params)
                .meanThresholds(mean)
                .belowThresholdMeans(below)
                .aboveThresholdMeans(above)
                .averageL2L1Ratio(l2l1Ratio)
                .rotationMatrix(rotationMatrix)
                .build();

        byte[] serialized = state.toByteArray();
        OneBitScalarQuantizationState deserialized = OneBitScalarQuantizationState.fromByteArray(serialized);

        assertArrayEquals(mean, deserialized.getMeanThresholds(), 0.0f);
        assertArrayEquals(below, deserialized.getBelowThresholdMeans(), 0.0f);
        assertArrayEquals(above, deserialized.getAboveThresholdMeans(), 0.0f);
        assertEquals(l2l1Ratio, deserialized.getAverageL2L1Ratio(), 1e-6);
        assertEquals(params, deserialized.getQuantizationParams());
        assertArrayEquals(rotationMatrix[0], deserialized.getRotationMatrix()[0], 0.0f);
    }

    public void testSerializeAndDeserializeMultiBitScalarQuantizationState() throws IOException {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT);
        float[][] thresholds = new float[][] {
                { 0.1f, 0.2f, 0.3f },
                { 0.4f, 0.5f, 0.6f }
        };
        float[] belowMeans = new float[] { 0.1f, 0.2f, 0.3f };
        float[] aboveMeans = new float[] { 0.5f, 0.6f, 0.7f };
        double l2l1Ratio = 0.65;
        float[][] rotationMatrix = new float[][] {
                { 1.0f, 0.0f, 0.0f },
                { 0.0f, 1.0f, 0.0f },
                { 0.0f, 0.0f, 1.0f }
        };

        MultiBitScalarQuantizationState state = MultiBitScalarQuantizationState.builder()
                .quantizationParams(params)
                .thresholds(thresholds)
                .belowThresholdMeans(belowMeans)
                .aboveThresholdMeans(aboveMeans)
                .averageL2L1Ratio(l2l1Ratio)
                .rotationMatrix(rotationMatrix)
                .build();

        byte[] serialized = state.toByteArray();
        MultiBitScalarQuantizationState deserialized = MultiBitScalarQuantizationState.fromByteArray(serialized);

        for (int i = 0; i < thresholds.length; i++) {
            assertArrayEquals(thresholds[i], deserialized.getThresholds()[i], 0.0f);
        }
        assertArrayEquals(belowMeans, deserialized.getBelowThresholdMeans(), 0.0f);
        assertArrayEquals(aboveMeans, deserialized.getAboveThresholdMeans(), 0.0f);
        assertEquals(l2l1Ratio, deserialized.getAverageL2L1Ratio(), 1e-6);
        assertEquals(params, deserialized.getQuantizationParams());
        assertArrayEquals(rotationMatrix[1], deserialized.getRotationMatrix()[1], 0.0f);
    }
}

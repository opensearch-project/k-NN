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
        ScalarQuantizationParams params = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();
        float[] mean = new float[] { 0.1f, 0.2f, 0.3f };
        float[][] rotationMatrix = new float[][] { { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } };

        OneBitScalarQuantizationState state = OneBitScalarQuantizationState.builder()
            .quantizationParams(params)
            .meanThresholds(mean)
            .rotationMatrix(rotationMatrix)
            .build();

        byte[] serialized = state.toByteArray();
        OneBitScalarQuantizationState deserialized = OneBitScalarQuantizationState.fromByteArray(serialized);

        assertArrayEquals(mean, deserialized.getMeanThresholds(), 0.0f);
        assertEquals(params, deserialized.getQuantizationParams());
        assertArrayEquals(rotationMatrix[0], deserialized.getRotationMatrix()[0], 0.0f);
    }

    public void testSerializeAndDeserializeOneBitScalarQuantizationStateWithADC() throws IOException {
        ScalarQuantizationParams params = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();
        float[] mean = new float[] { 0.1f, 0.2f, 0.3f };
        float[][] rotationMatrix = new float[][] { { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } };
        float[] above = new float[] { 0.2f, 0.3f, 0.4f };
        float[] below = new float[] { 0.0f, 0.1f, 0.2f };
        OneBitScalarQuantizationState state = OneBitScalarQuantizationState.builder()
                .quantizationParams(params)
                .meanThresholds(mean)
                .aboveThresholdMeans(above)
                .belowThresholdMeans(below)
                .rotationMatrix(rotationMatrix)
                .build();

        byte[] serialized = state.toByteArray();
        OneBitScalarQuantizationState deserialized = OneBitScalarQuantizationState.fromByteArray(serialized);

        assertArrayEquals(mean, deserialized.getMeanThresholds(), 0.0f);
        assertArrayEquals(above, deserialized.getAboveThresholdMeans(), 0.0f);
        assertArrayEquals(below, deserialized.getBelowThresholdMeans(), 0.0f);
        assertArrayEquals(rotationMatrix[2], deserialized.getRotationMatrix()[2], 0.0f);
        assertEquals(params, deserialized.getQuantizationParams());
    }

    public void testSerializeAndDeserializeMultiBitScalarQuantizationState() throws IOException {
        ScalarQuantizationParams params = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.TWO_BIT).build();
        float[][] thresholds = new float[][] { { 0.1f, 0.2f, 0.3f }, { 0.4f, 0.5f, 0.6f } };
        float[][] rotationMatrix = new float[][] { { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } };

        MultiBitScalarQuantizationState state = MultiBitScalarQuantizationState.builder()
            .quantizationParams(params)
            .thresholds(thresholds)
            .rotationMatrix(rotationMatrix)
            .build();

        byte[] serialized = state.toByteArray();
        MultiBitScalarQuantizationState deserialized = MultiBitScalarQuantizationState.fromByteArray(serialized);

        for (int i = 0; i < thresholds.length; i++) {
            assertArrayEquals(thresholds[i], deserialized.getThresholds()[i], 0.0f);
        }
        assertEquals(params, deserialized.getQuantizationParams());
        assertArrayEquals(rotationMatrix[1], deserialized.getRotationMatrix()[1], 0.0f);
    }
}

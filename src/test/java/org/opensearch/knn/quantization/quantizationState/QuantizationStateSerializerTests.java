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
        OneBitScalarQuantizationState state = OneBitScalarQuantizationState.builder()
            .quantizationParams(params)
            .meanThresholds(mean)
            .build();

        // Serialize
        byte[] serialized = state.toByteArray();

        OneBitScalarQuantizationState deserialized = OneBitScalarQuantizationState.fromByteArray(serialized);

        assertArrayEquals(mean, deserialized.getMeanThresholds(), 0.0f);
        assertEquals(params, deserialized.getQuantizationParams());
    }

    public void testSerializeAndDeserializeMultiBitScalarQuantizationState() throws IOException {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT);
        float[][] thresholds = new float[][] { { 0.1f, 0.2f, 0.3f }, { 0.4f, 0.5f, 0.6f } };
        MultiBitScalarQuantizationState state = new MultiBitScalarQuantizationState(params, thresholds);

        // Serialize
        byte[] serialized = state.toByteArray();
        MultiBitScalarQuantizationState deserialized = MultiBitScalarQuantizationState.fromByteArray(serialized);

        for (int i = 0; i < thresholds.length; i++) {
            assertArrayEquals(thresholds[i], deserialized.getThresholds()[i], 0.0f);
        }
        assertEquals(params, deserialized.getQuantizationParams());
    }
}

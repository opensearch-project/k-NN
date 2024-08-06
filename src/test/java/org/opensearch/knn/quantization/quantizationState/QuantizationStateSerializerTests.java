/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizationState;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.SQParams;
import org.opensearch.knn.quantization.models.quantizationState.MultiBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;

import java.io.IOException;

public class QuantizationStateSerializerTests extends KNNTestCase {

    public void testSerializeAndDeserializeOneBitScalarQuantizationState() throws IOException, ClassNotFoundException {
        SQParams params = new SQParams(ScalarQuantizationType.ONE_BIT);
        float[] mean = new float[] { 0.1f, 0.2f, 0.3f };
        OneBitScalarQuantizationState state = new OneBitScalarQuantizationState(params, mean);

        byte[] serialized = state.toByteArray();
        OneBitScalarQuantizationState deserialized = OneBitScalarQuantizationState.fromByteArray(serialized);

        assertArrayEquals(mean, deserialized.getMeanThresholds(), 0.0f);
        assertEquals(params, deserialized.getQuantizationParams());
    }

    public void testSerializeAndDeserializeMultiBitScalarQuantizationState() throws IOException, ClassNotFoundException {
        SQParams params = new SQParams(ScalarQuantizationType.TWO_BIT);
        float[][] thresholds = new float[][] { { 0.1f, 0.2f, 0.3f }, { 0.4f, 0.5f, 0.6f } };
        MultiBitScalarQuantizationState state = new MultiBitScalarQuantizationState(params, thresholds);

        byte[] serialized = state.toByteArray();
        MultiBitScalarQuantizationState deserialized = MultiBitScalarQuantizationState.fromByteArray(serialized);

        assertArrayEquals(thresholds, deserialized.getThresholds());
        assertEquals(params, deserialized.getQuantizationParams());
    }
}

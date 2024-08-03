/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 */

package org.opensearch.knn.quantization.quantizationState;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.SQParams;
import org.opensearch.knn.quantization.models.quantizationState.DefaultQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.MultiBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;

import java.io.IOException;

public class QuantizationStateTests extends KNNTestCase {

    public void testOneBitScalarQuantizationStateSerialization() throws IOException, ClassNotFoundException {
        SQParams params = new SQParams(ScalarQuantizationType.ONE_BIT);
        float[] mean = { 1.0f, 2.0f, 3.0f };

        OneBitScalarQuantizationState state = new OneBitScalarQuantizationState(params, mean);

        byte[] serializedState = state.toByteArray();
        OneBitScalarQuantizationState deserializedState = OneBitScalarQuantizationState.fromByteArray(serializedState);
        float delta = 0.0001f;

        assertArrayEquals(mean, deserializedState.getMeanThresholds(), delta);
        assertEquals(params.getQuantizationType(), deserializedState.getQuantizationParams().getQuantizationType());
    }

    public void testMultiBitScalarQuantizationStateSerialization() throws IOException, ClassNotFoundException {
        SQParams params = new SQParams(ScalarQuantizationType.TWO_BIT);
        float[][] thresholds = { { 0.5f, 1.5f, 2.5f }, { 1.0f, 2.0f, 3.0f } };

        MultiBitScalarQuantizationState state = new MultiBitScalarQuantizationState(params, thresholds);

        byte[] serializedState = state.toByteArray();
        MultiBitScalarQuantizationState deserializedState = MultiBitScalarQuantizationState.fromByteArray(serializedState);
        float delta = 0.0001f;

        for (int i = 0; i < thresholds.length; i++) {
            assertArrayEquals(thresholds[i], deserializedState.getThresholds()[i], delta);
        }
        assertEquals(params.getQuantizationType(), deserializedState.getQuantizationParams().getQuantizationType());
    }

    public void testDefaultQuantizationStateSerialization() throws IOException, ClassNotFoundException {
        SQParams params = new SQParams(ScalarQuantizationType.UNSUPPORTED_TYPE);

        DefaultQuantizationState state = new DefaultQuantizationState(params);

        byte[] serializedState = state.toByteArray();
        DefaultQuantizationState deserializedState = DefaultQuantizationState.fromByteArray(serializedState);

        assertEquals(params.getQuantizationType(), deserializedState.getQuantizationParams().getQuantizationType());
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 */

package org.opensearch.knn.quantization.quantizationState;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.MultiBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;
import org.opensearch.Version;
import org.opensearch.knn.quantization.util.VersionContext;

import java.io.IOException;

public class QuantizationStateTests extends KNNTestCase {

    public void testOneBitScalarQuantizationStateSerialization() throws IOException, ClassNotFoundException {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        float[] mean = { 1.0f, 2.0f, 3.0f };

        OneBitScalarQuantizationState state = new OneBitScalarQuantizationState(params, mean);

        // Set the version for serialization
        VersionContext.setVersion(Version.CURRENT.id);

        // Serialize
        byte[] serializedState = state.toByteArray();

        // Deserialize
        OneBitScalarQuantizationState deserializedState = OneBitScalarQuantizationState.fromByteArray(serializedState);

        float delta = 0.0001f;
        assertArrayEquals(mean, deserializedState.getMeanThresholds(), delta);
        assertEquals(params.getSqType(), deserializedState.getQuantizationParams().getSqType());
    }

    public void testMultiBitScalarQuantizationStateSerialization() throws IOException, ClassNotFoundException {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT);
        float[][] thresholds = { { 0.5f, 1.5f, 2.5f }, { 1.0f, 2.0f, 3.0f } };

        MultiBitScalarQuantizationState state = new MultiBitScalarQuantizationState(params, thresholds);

        // Set the version for serialization
        VersionContext.setVersion(Version.CURRENT.id);

        // Serialize
        byte[] serializedState = state.toByteArray();

        // Deserialize
        MultiBitScalarQuantizationState deserializedState = MultiBitScalarQuantizationState.fromByteArray(serializedState);

        float delta = 0.0001f;
        for (int i = 0; i < thresholds.length; i++) {
            assertArrayEquals(thresholds[i], deserializedState.getThresholds()[i], delta);
        }
        assertEquals(params.getSqType(), deserializedState.getQuantizationParams().getSqType());
    }

    public void testSerializationWithDifferentVersions() throws IOException, ClassNotFoundException {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        float[] mean = { 1.0f, 2.0f, 3.0f };

        OneBitScalarQuantizationState state = new OneBitScalarQuantizationState(params, mean);

        // Simulate an older version
        VersionContext.setVersion(Version.V_2_0_0.id);

        // Serialize
        byte[] serializedState = state.toByteArray();

        // Update to a new version and deserialize
        VersionContext.setVersion(Version.CURRENT.id);
        OneBitScalarQuantizationState deserializedState = OneBitScalarQuantizationState.fromByteArray(serializedState);

        float delta = 0.0001f;
        assertArrayEquals(mean, deserializedState.getMeanThresholds(), delta);
        assertEquals(params.getSqType(), deserializedState.getQuantizationParams().getSqType());
    }
}

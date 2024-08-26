/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizationState;

import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.MultiBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;

import java.io.IOException;

public class QuantizationStateTests extends KNNTestCase {

    public void testOneBitScalarQuantizationStateSerialization() throws IOException {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        float[] mean = { 1.0f, 2.0f, 3.0f };

        OneBitScalarQuantizationState state = new OneBitScalarQuantizationState(params, mean);

        // Serialize
        byte[] serializedState = state.toByteArray();

        // Deserialize
        StreamInput in = StreamInput.wrap(serializedState);
        OneBitScalarQuantizationState deserializedState = new OneBitScalarQuantizationState(in);

        float delta = 0.0001f;
        assertArrayEquals(mean, deserializedState.getMeanThresholds(), delta);
        assertEquals(params.getSqType(), deserializedState.getQuantizationParams().getSqType());
    }

    public void testMultiBitScalarQuantizationStateSerialization() throws IOException {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT);
        float[][] thresholds = { { 0.5f, 1.5f, 2.5f }, { 1.0f, 2.0f, 3.0f } };

        MultiBitScalarQuantizationState state = new MultiBitScalarQuantizationState(params, thresholds);
        byte[] serializedState = state.toByteArray();

        // Deserialize
        StreamInput in = StreamInput.wrap(serializedState);
        MultiBitScalarQuantizationState deserializedState = new MultiBitScalarQuantizationState(in);

        float delta = 0.0001f;
        for (int i = 0; i < thresholds.length; i++) {
            assertArrayEquals(thresholds[i], deserializedState.getThresholds()[i], delta);
        }
        assertEquals(params.getSqType(), deserializedState.getQuantizationParams().getSqType());
    }

    public void testSerializationWithDifferentVersions() throws IOException {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        float[] mean = { 1.0f, 2.0f, 3.0f };

        OneBitScalarQuantizationState state = new OneBitScalarQuantizationState(params, mean);
        byte[] serializedState = state.toByteArray();
        StreamInput in = StreamInput.wrap(serializedState);
        OneBitScalarQuantizationState deserializedState = new OneBitScalarQuantizationState(in);

        float delta = 0.0001f;
        assertArrayEquals(mean, deserializedState.getMeanThresholds(), delta);
        assertEquals(params.getSqType(), deserializedState.getQuantizationParams().getSqType());
    }

    public void testOneBitScalarQuantizationStateRamBytesUsed() throws IOException {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        float[] mean = { 1.0f, 2.0f, 3.0f };

        OneBitScalarQuantizationState state = new OneBitScalarQuantizationState(params, mean);

        // 1. Manual Calculation of RAM Usage
        long manualEstimatedRamBytesUsed = 0L;

        // OneBitScalarQuantizationState object overhead for Object Header
        manualEstimatedRamBytesUsed += alignSize(16L);

        // ScalarQuantizationParams object overhead Object Header
        manualEstimatedRamBytesUsed += alignSize(16L);

        // Mean array overhead (array header + size of elements)
        manualEstimatedRamBytesUsed += alignSize(16L + 4L * mean.length);

        // 3. RAM Usage from RamUsageEstimator
        long expectedRamBytesUsed = RamUsageEstimator.shallowSizeOfInstance(OneBitScalarQuantizationState.class) + RamUsageEstimator
            .shallowSizeOf(params) + RamUsageEstimator.sizeOf(mean);

        long actualRamBytesUsed = state.ramBytesUsed();

        // Allow a difference between manual estimation, serialization size, and actual RAM usage
        assertTrue(
            "The difference between manual and actual RAM usage exceeds 8 bytes",
            Math.abs(manualEstimatedRamBytesUsed - actualRamBytesUsed) <= 8
        );

        assertEquals(expectedRamBytesUsed, actualRamBytesUsed);
    }

    public void testMultiBitScalarQuantizationStateRamBytesUsedManualCalculation() throws IOException {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT);
        float[][] thresholds = { { 0.5f, 1.5f, 2.5f }, { 1.0f, 2.0f, 3.0f } };

        MultiBitScalarQuantizationState state = new MultiBitScalarQuantizationState(params, thresholds);

        // Manually estimate RAM usage with alignment
        long manualEstimatedRamBytesUsed = 0L;

        // Estimate for MultiBitScalarQuantizationState object
        manualEstimatedRamBytesUsed += alignSize(16L);  // Example overhead for object

        // Estimate for ScalarQuantizationParams object
        manualEstimatedRamBytesUsed += alignSize(16L);  // Overhead for params object (including fields)

        // Estimate for thresholds array
        manualEstimatedRamBytesUsed += alignSize(16L + 4L * thresholds.length);  // Overhead for array + references to sub-arrays

        for (float[] row : thresholds) {
            manualEstimatedRamBytesUsed += alignSize(16L + 4L * row.length);  // Overhead for each sub-array + size of each float
        }

        long ramEstimatorRamBytesUsed = RamUsageEstimator.shallowSizeOfInstance(MultiBitScalarQuantizationState.class) + RamUsageEstimator
            .shallowSizeOf(params) + RamUsageEstimator.shallowSizeOf(thresholds);

        for (float[] row : thresholds) {
            ramEstimatorRamBytesUsed += RamUsageEstimator.sizeOf(row);
        }

        long difference = Math.abs(manualEstimatedRamBytesUsed - ramEstimatorRamBytesUsed);
        assertTrue("The difference between manual and actual RAM usage exceeds 8 bytes", difference <= 8);
        assertEquals(ramEstimatorRamBytesUsed, state.ramBytesUsed());
    }

    private long alignSize(long size) {
        return (size + 7) & ~7;  // Align to 8 bytes boundary
    }

}

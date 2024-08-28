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

    public void testMultiBitScalarQuantizationStateGetDimensionsWithDimensionNotMultipleOf8() {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT);

        // Case 1: 3 thresholds, each with 2 dimensions
        float[][] thresholds1 = { { 0.5f, 1.5f }, { 1.0f, 2.0f }, { 1.5f, 2.5f } };
        MultiBitScalarQuantizationState state1 = new MultiBitScalarQuantizationState(params, thresholds1);
        int expectedDimensions1 = 24; // The next multiple of 8 considering all bits
        assertEquals(expectedDimensions1, state1.getDimensions());

        // Case 2: 1 threshold, with 5 dimensions (5 bits, should align to 8)
        float[][] thresholds2 = { { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f } };
        MultiBitScalarQuantizationState state2 = new MultiBitScalarQuantizationState(params, thresholds2);
        int expectedDimensions2 = 8; // The next multiple of 8 considering all bits
        assertEquals(expectedDimensions2, state2.getDimensions());

        // Case 3: 4 thresholds, each with 7 dimensions (28 bits, should align to 32)
        float[][] thresholds3 = {
            { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f },
            { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f },
            { 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f },
            { 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f } };
        MultiBitScalarQuantizationState state3 = new MultiBitScalarQuantizationState(params, thresholds3);
        int expectedDimensions3 = 32; // The next multiple of 8 considering all bits
        assertEquals(expectedDimensions3, state3.getDimensions());

        // Case 4: 2 thresholds, each with 8 dimensions (16 bits, already aligned)
        float[][] thresholds4 = { { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f }, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f } };
        MultiBitScalarQuantizationState state4 = new MultiBitScalarQuantizationState(params, thresholds4);
        int expectedDimensions4 = 16; // Already aligned to 8
        assertEquals(expectedDimensions4, state4.getDimensions());

        // Case 5: 2 thresholds, each with 6 dimensions (12 bits, should align to 16)
        float[][] thresholds5 = { { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f }, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f } };
        MultiBitScalarQuantizationState state5 = new MultiBitScalarQuantizationState(params, thresholds5);
        int expectedDimensions5 = 16; // The next multiple of 8 considering all bits
        assertEquals(expectedDimensions5, state5.getDimensions());
    }

    public void testOneBitScalarQuantizationStateGetDimensionsWithDimensionNotMultipleOf8() {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);

        // Case 1: 5 dimensions (should align to 8)
        float[] thresholds1 = { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f };
        OneBitScalarQuantizationState state1 = new OneBitScalarQuantizationState(params, thresholds1);
        int expectedDimensions1 = 8; // The next multiple of 8
        assertEquals(expectedDimensions1, state1.getDimensions());

        // Case 2: 7 dimensions (should align to 8)
        float[] thresholds2 = { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f };
        OneBitScalarQuantizationState state2 = new OneBitScalarQuantizationState(params, thresholds2);
        int expectedDimensions2 = 8; // The next multiple of 8
        assertEquals(expectedDimensions2, state2.getDimensions());

        // Case 3: 8 dimensions (already aligned to 8)
        float[] thresholds3 = { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f };
        OneBitScalarQuantizationState state3 = new OneBitScalarQuantizationState(params, thresholds3);
        int expectedDimensions3 = 8; // Already aligned to 8
        assertEquals(expectedDimensions3, state3.getDimensions());

        // Case 4: 10 dimensions (should align to 16)
        float[] thresholds4 = { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f, 9.5f };
        OneBitScalarQuantizationState state4 = new OneBitScalarQuantizationState(params, thresholds4);
        int expectedDimensions4 = 16; // The next multiple of 8
        assertEquals(expectedDimensions4, state4.getDimensions());

        // Case 5: 16 dimensions (already aligned to 16)
        float[] thresholds5 = { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f, 9.5f, 10.5f, 11.5f, 12.5f, 13.5f, 14.5f, 15.5f };
        OneBitScalarQuantizationState state5 = new OneBitScalarQuantizationState(params, thresholds5);
        int expectedDimensions5 = 16; // Already aligned to 16
        assertEquals(expectedDimensions5, state5.getDimensions());
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

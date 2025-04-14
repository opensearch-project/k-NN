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

        OneBitScalarQuantizationState state = OneBitScalarQuantizationState.builder()
            .quantizationParams(params)
            .meanThresholds(mean)
            .build();

        // Serialize
        byte[] serializedState = state.toByteArray();

        // Deserialize
        StreamInput in = StreamInput.wrap(serializedState);
        OneBitScalarQuantizationState deserializedState = new OneBitScalarQuantizationState(in);

        float delta = 0.0001f;
        assertArrayEquals(mean, deserializedState.getMeanThresholds(), delta);
        assertEquals(params.getSqType(), deserializedState.getQuantizationParams().getSqType());
        assertNull(deserializedState.getBelowThresholdMeans());
        assertNull(deserializedState.getAboveThresholdMeans());
    }

    // Test serialization and deserialization with optional fields
    public void testOneBitScalarQuantizationState_WithOptionalFields() throws IOException {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        float[] mean = { 1.0f, 2.0f, 3.0f };
        float[] belowThresholdMeans = { 0.5f, 1.5f, 2.5f };
        float[] aboveThresholdMeans = { 1.5f, 2.5f, 3.5f };

        OneBitScalarQuantizationState state = OneBitScalarQuantizationState.builder()
            .quantizationParams(params)
            .meanThresholds(mean)
            .aboveThresholdMeans(aboveThresholdMeans)
            .belowThresholdMeans(belowThresholdMeans)
            .build();

        // Serialize
        byte[] serializedState = state.toByteArray();

        // Deserialize
        StreamInput in = StreamInput.wrap(serializedState);
        OneBitScalarQuantizationState deserializedState = new OneBitScalarQuantizationState(in);

        // Assertions
        assertArrayEquals(mean, deserializedState.getMeanThresholds(), 0.0f);
        assertArrayEquals(belowThresholdMeans, deserializedState.getBelowThresholdMeans(), 0.0f);
        assertArrayEquals(aboveThresholdMeans, deserializedState.getAboveThresholdMeans(), 0.0f);
        assertEquals(params, deserializedState.getQuantizationParams());
    }

    // Test handling of null arrays in RAM usage
    public void testOneBitScalarQuantizationState_RamBytesUsedWithNulls() {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        float[] mean = { 1.0f, 2.0f, 3.0f };

        OneBitScalarQuantizationState state = OneBitScalarQuantizationState.builder()
            .quantizationParams(params)
            .meanThresholds(mean)
            .build();

        long actualRamBytesUsed = state.ramBytesUsed();
        long expectedRamBytesUsed = RamUsageEstimator.shallowSizeOfInstance(OneBitScalarQuantizationState.class) + RamUsageEstimator
            .shallowSizeOf(params) + RamUsageEstimator.sizeOf(mean);

        assertEquals(expectedRamBytesUsed, actualRamBytesUsed);
    }

    // Test handling of all fields in RAM usage
    public void testOneBitScalarQuantizationState_RamBytesUsedWithAllFields() {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        float[] mean = { 1.0f, 2.0f, 3.0f };
        float[] belowThresholdMeans = { 0.5f, 1.5f, 2.5f };
        float[] aboveThresholdMeans = { 1.5f, 2.5f, 3.5f };

        OneBitScalarQuantizationState state = OneBitScalarQuantizationState.builder()
            .quantizationParams(params)
            .meanThresholds(mean)
            .aboveThresholdMeans(aboveThresholdMeans)
            .belowThresholdMeans(belowThresholdMeans)
            .build();

        long actualRamBytesUsed = state.ramBytesUsed();
        long expectedRamBytesUsed = RamUsageEstimator.shallowSizeOfInstance(OneBitScalarQuantizationState.class) + RamUsageEstimator
            .shallowSizeOf(params) + RamUsageEstimator.sizeOf(mean) + RamUsageEstimator.sizeOf(belowThresholdMeans) + RamUsageEstimator
                .sizeOf(aboveThresholdMeans);

        assertEquals(expectedRamBytesUsed, actualRamBytesUsed);
    }

    public void testMultiBitScalarQuantizationStateSerialization() throws IOException {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT);
        float[][] thresholds = { { 0.5f, 1.5f, 2.5f }, { 1.0f, 2.0f, 3.0f } };

        MultiBitScalarQuantizationState state = MultiBitScalarQuantizationState.builder()
                .quantizationParams(params)
                .thresholds(thresholds)
                .build();

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

    public void testMultiBitScalarQuantizationStateRamBytesUsedManualCalculation() throws IOException {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT);
        float[][] thresholds = { { 0.5f, 1.5f, 2.5f }, { 1.0f, 2.0f, 3.0f } };

        MultiBitScalarQuantizationState state = MultiBitScalarQuantizationState.builder()
                .quantizationParams(params)
                .thresholds(thresholds)
                .build();

        long manualEstimatedRamBytesUsed = 0L;
        manualEstimatedRamBytesUsed += alignSize(16L); // object overhead
        manualEstimatedRamBytesUsed += alignSize(16L); // param object
        manualEstimatedRamBytesUsed += alignSize(16L + 4L * thresholds.length);
        manualEstimatedRamBytesUsed += alignSize(16L); // for Above and below Threshold for Object Oveerhead
        for (float[] row : thresholds) {
            manualEstimatedRamBytesUsed += alignSize(16L + 4L * row.length);
        }

        long ramEstimatorRamBytesUsed = RamUsageEstimator.shallowSizeOfInstance(MultiBitScalarQuantizationState.class)
                + RamUsageEstimator.shallowSizeOf(params)
                + RamUsageEstimator.shallowSizeOf(thresholds);

        for (float[] row : thresholds) {
            ramEstimatorRamBytesUsed += RamUsageEstimator.sizeOf(row);
        }

        long difference = Math.abs(manualEstimatedRamBytesUsed - ramEstimatorRamBytesUsed);
        assertTrue("RAM usage difference too high", difference <= 8);
        assertEquals(ramEstimatorRamBytesUsed, state.ramBytesUsed());
    }

    public void testMultiBitScalarQuantizationStateGetDimensions_withAlignedThresholds() {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT);
        float[][] thresholds = {
                { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f },
                { 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f }
        };

        MultiBitScalarQuantizationState state = MultiBitScalarQuantizationState.builder()
                .quantizationParams(params)
                .thresholds(thresholds)
                .build();

        int expectedDimensions = 16; // 2 bit levels × 8 dims already aligned
        assertEquals(expectedDimensions, state.getDimensions());
    }

    public void testMultiBitScalarQuantizationStateGetDimensions_withUnalignedThresholds() {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT);
        float[][] thresholds = {
                { 0.1f, 0.2f, 0.3f },
                { 1.1f, 1.2f, 1.3f }
        };

        MultiBitScalarQuantizationState state = MultiBitScalarQuantizationState.builder()
                .quantizationParams(params)
                .thresholds(thresholds)
                .build();

        int expectedDimensions = 16; // 2 bits × 3 dims = 6 bits, padded to next multiple of 8 = 16
        assertEquals(expectedDimensions, state.getDimensions());
    }



    public void testOneBitScalarQuantizationStateGetDimensions_withDimensionNotMultipleOf8_thenSuccess() {
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);

        // Case 1: 5 dimensions (should align to 8)
        float[] thresholds1 = { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f };
        OneBitScalarQuantizationState state1 = OneBitScalarQuantizationState.builder()
            .quantizationParams(params)
            .meanThresholds(thresholds1)
            .build();
        int expectedDimensions1 = 8; // The next multiple of 8
        assertEquals(expectedDimensions1, state1.getDimensions());

        // Case 2: 7 dimensions (should align to 8)
        float[] thresholds2 = { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f };
        OneBitScalarQuantizationState state2 = OneBitScalarQuantizationState.builder()
            .quantizationParams(params)
            .meanThresholds(thresholds2)
            .build();
        int expectedDimensions2 = 8; // The next multiple of 8
        assertEquals(expectedDimensions2, state2.getDimensions());

        // Case 3: 8 dimensions (already aligned to 8)
        float[] thresholds3 = { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f };
        OneBitScalarQuantizationState state3 = OneBitScalarQuantizationState.builder()
            .quantizationParams(params)
            .meanThresholds(thresholds3)
            .build();
        int expectedDimensions3 = 8; // Already aligned to 8
        assertEquals(expectedDimensions3, state3.getDimensions());

        // Case 4: 10 dimensions (should align to 16)
        float[] thresholds4 = { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f, 9.5f };
        OneBitScalarQuantizationState state4 = OneBitScalarQuantizationState.builder()
            .quantizationParams(params)
            .meanThresholds(thresholds4)
            .build();
        int expectedDimensions4 = 16; // The next multiple of 8
        assertEquals(expectedDimensions4, state4.getDimensions());

        // Case 5: 16 dimensions (already aligned to 16)
        float[] thresholds5 = { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f, 9.5f, 10.5f, 11.5f, 12.5f, 13.5f, 14.5f, 15.5f };
        OneBitScalarQuantizationState state5 = OneBitScalarQuantizationState.builder()
            .quantizationParams(params)
            .meanThresholds(thresholds5)
            .build();
        int expectedDimensions5 = 16; // Already aligned to 16
        assertEquals(expectedDimensions5, state5.getDimensions());
    }

    private long alignSize(long size) {
        return (size + 7) & ~7;  // Align to 8 bytes boundary
    }

}

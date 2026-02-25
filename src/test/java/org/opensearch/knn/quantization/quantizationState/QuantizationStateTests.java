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
        ScalarQuantizationParams params = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();
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
    }

    // Test serialization and deserialization with optional fields
    public void testOneBitScalarQuantizationState_WithOptionalFields() throws IOException {
        ScalarQuantizationParams params = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();
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

        // Assertions
        assertArrayEquals(mean, deserializedState.getMeanThresholds(), 0.0f);
        assertEquals(params, deserializedState.getQuantizationParams());
    }

    // Test handling of null arrays in RAM usage
    public void testOneBitScalarQuantizationState_RamBytesUsedWithNulls() {
        ScalarQuantizationParams params = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();
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
        ScalarQuantizationParams params = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();
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

    public void testOneBitScalarQuantizationState_RamBytesUsedWithAboveBelowThresholds() {
        ScalarQuantizationParams params = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();
        float[] mean = { 1.0f, 2.0f, 3.0f };
        float[] above = { 1.0f, 2.0f, 3.0f };
        float[] below = { 1.0f, 2.0f, 3.0f };

        OneBitScalarQuantizationState state = OneBitScalarQuantizationState.builder()
            .quantizationParams(params)
            .meanThresholds(mean)
            .aboveThresholdMeans(above)
            .belowThresholdMeans(below)
            .build();

        long actualRamBytesUsed = state.ramBytesUsed();
        long expectedRamBytesUsed = RamUsageEstimator.shallowSizeOfInstance(OneBitScalarQuantizationState.class) + RamUsageEstimator
            .shallowSizeOf(params) + RamUsageEstimator.sizeOf(mean) + RamUsageEstimator.sizeOf(above) + RamUsageEstimator.sizeOf(below);

        assertEquals(expectedRamBytesUsed, actualRamBytesUsed);
    }

    public void testMultiBitScalarQuantizationStateSerialization() throws IOException {
        ScalarQuantizationParams params = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.TWO_BIT).build();
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
        ScalarQuantizationParams params = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.TWO_BIT).build();
        float[][] thresholds = { { 0.5f, 1.5f, 2.5f }, { 1.0f, 2.0f, 3.0f } };

        MultiBitScalarQuantizationState state = MultiBitScalarQuantizationState.builder()
            .quantizationParams(params)
            .thresholds(thresholds)
            .build();

        long manualEstimatedRamBytesUsed = 0L;
        manualEstimatedRamBytesUsed += alignSize(16L); // object overhead
        manualEstimatedRamBytesUsed += alignSize(24L); // param object (sqType + isRandomRotation + isEnableADC)
        manualEstimatedRamBytesUsed += alignSize(16L + 4L * thresholds.length);
        for (float[] row : thresholds) {
            manualEstimatedRamBytesUsed += alignSize(16L + 4L * row.length);
        }

        long ramEstimatorRamBytesUsed = RamUsageEstimator.shallowSizeOfInstance(MultiBitScalarQuantizationState.class) + RamUsageEstimator
            .shallowSizeOf(params) + RamUsageEstimator.shallowSizeOf(thresholds);

        for (float[] row : thresholds) {
            ramEstimatorRamBytesUsed += RamUsageEstimator.sizeOf(row);
        }

        long difference = Math.abs(manualEstimatedRamBytesUsed - ramEstimatorRamBytesUsed);
        assertTrue("RAM usage difference too high", difference <= 8);
        assertEquals(ramEstimatorRamBytesUsed, state.ramBytesUsed());
    }

    public void testMultiBitScalarQuantizationStateGetDimensions_withAlignedThresholds() {
        ScalarQuantizationParams params = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.TWO_BIT).build();
        float[][] thresholds = { { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f }, { 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f } };

        MultiBitScalarQuantizationState state = MultiBitScalarQuantizationState.builder()
            .quantizationParams(params)
            .thresholds(thresholds)
            .build();

        int expectedDimensions = 16; // 2 bit levels × 8 dims already aligned
        assertEquals(expectedDimensions, state.getDimensions());
    }

    public void testMultiBitScalarQuantizationStateGetDimensions_withUnalignedThresholds() {
        ScalarQuantizationParams params = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.TWO_BIT).build();

        // Case 1: 3D with 2 bits: 3*2=6 bits → align to 8 bits
        float[][] thresholds = { { 0.1f, 0.2f, 0.3f }, { 1.1f, 1.2f, 1.3f } };
        MultiBitScalarQuantizationState state = MultiBitScalarQuantizationState.builder()
            .quantizationParams(params)
            .thresholds(thresholds)
            .build();
        int expectedDimensions = 8; // 6 bits aligned to 8
        assertEquals(expectedDimensions, state.getDimensions());

        // Case 2: 2D with 3 bit levels: 2*3=6 bits → align to 8 bits
        float[][] thresholds1 = { { 0.5f, 1.5f }, { 1.0f, 2.0f }, { 1.5f, 2.5f } };
        MultiBitScalarQuantizationState state1 = new MultiBitScalarQuantizationState(params, thresholds1, null);
        int expectedDimensions1 = 8; // 6 bits aligned to 8
        assertEquals(expectedDimensions1, state1.getDimensions());
    }

    public void testMultiBitScalarQuantizationState_getBytesPerVector() {
        // Test 2D with 2 bits (16x compression)
        ScalarQuantizationParams params2bit = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.TWO_BIT).build();
        float[][] thresholds2D = { { 0.5f, 1.5f }, { 1.0f, 2.0f } };
        MultiBitScalarQuantizationState state2D = MultiBitScalarQuantizationState.builder()
            .quantizationParams(params2bit)
            .thresholds(thresholds2D)
            .build();
        assertEquals(1, state2D.getBytesPerVector()); // 4 bits = 1 byte

        // Test 12D with 2 bits (16x compression)
        float[][] thresholds12D = new float[2][12];
        MultiBitScalarQuantizationState state12D = MultiBitScalarQuantizationState.builder()
            .quantizationParams(params2bit)
            .thresholds(thresholds12D)
            .build();
        assertEquals(3, state12D.getBytesPerVector()); // 24 bits = 3 bytes

        // Test 2D with 4 bits (8x compression)
        ScalarQuantizationParams params4bit = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.FOUR_BIT).build();
        float[][] thresholds2D_4bit = new float[4][2];
        MultiBitScalarQuantizationState state2D_4bit = MultiBitScalarQuantizationState.builder()
            .quantizationParams(params4bit)
            .thresholds(thresholds2D_4bit)
            .build();
        assertEquals(1, state2D_4bit.getBytesPerVector()); // 8 bits = 1 byte

        // Test 12D with 4 bits (8x compression)
        float[][] thresholds12D_4bit = new float[4][12];
        MultiBitScalarQuantizationState state12D_4bit = MultiBitScalarQuantizationState.builder()
            .quantizationParams(params4bit)
            .thresholds(thresholds12D_4bit)
            .build();
        assertEquals(6, state12D_4bit.getBytesPerVector()); // 48 bits = 6 bytes
    }

    public void testOneBitScalarQuantizationState_getBytesPerVector() {
        ScalarQuantizationParams params = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();

        // Test 2D with 1 bit (32x compression)
        float[] thresholds2D = { 0.5f, 1.5f };
        OneBitScalarQuantizationState state2D = OneBitScalarQuantizationState.builder()
            .quantizationParams(params)
            .meanThresholds(thresholds2D)
            .build();
        assertEquals(1, state2D.getBytesPerVector()); // 2 bits = 1 byte

        // Test 12D with 1 bit (32x compression)
        float[] thresholds12D = new float[12];
        OneBitScalarQuantizationState state12D = OneBitScalarQuantizationState.builder()
            .quantizationParams(params)
            .meanThresholds(thresholds12D)
            .build();
        assertEquals(2, state12D.getBytesPerVector()); // 12 bits = 2 bytes
    }

    public void testMultiBitScalarQuantizationState_getDimensions_fixedLogic() {
        // Test 2D with 2 bits: 2*2=4 bits → align to 8 bits
        ScalarQuantizationParams params2bit = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.TWO_BIT).build();
        float[][] thresholds2D = { { 0.5f, 1.5f }, { 1.0f, 2.0f } };
        MultiBitScalarQuantizationState state2D = MultiBitScalarQuantizationState.builder()
            .quantizationParams(params2bit)
            .thresholds(thresholds2D)
            .build();
        assertEquals(8, state2D.getDimensions()); // 4 bits aligned to 8

        // Test 12D with 2 bits: 12*2=24 bits → already aligned
        float[][] thresholds12D = new float[2][12];
        MultiBitScalarQuantizationState state12D = MultiBitScalarQuantizationState.builder()
            .quantizationParams(params2bit)
            .thresholds(thresholds12D)
            .build();
        assertEquals(24, state12D.getDimensions()); // 24 bits already aligned

        // Test 5D with 4 bits: 5*4=20 bits → align to 24 bits
        ScalarQuantizationParams params4bit = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.FOUR_BIT).build();
        float[][] thresholds5D_4bit = new float[4][5];
        MultiBitScalarQuantizationState state5D_4bit = MultiBitScalarQuantizationState.builder()
            .quantizationParams(params4bit)
            .thresholds(thresholds5D_4bit)
            .build();
        assertEquals(24, state5D_4bit.getDimensions()); // 20 bits aligned to 24

        // Test 12D with 4 bits: 12*4=48 bits → already aligned
        float[][] thresholds12D_4bit = new float[4][12];
        MultiBitScalarQuantizationState state12D_4bit = MultiBitScalarQuantizationState.builder()
            .quantizationParams(params4bit)
            .thresholds(thresholds12D_4bit)
            .build();
        assertEquals(48, state12D_4bit.getDimensions()); // 48 bits already aligned
    }

    public void testOneBitScalarQuantizationStateGetDimensions_withDimensionNotMultipleOf8_thenSuccess() {
        ScalarQuantizationParams params = ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();

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

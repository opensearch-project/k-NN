/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.opensearch.knn.common.annotation.ExpectRemoteBuildValidation;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

/**
 * This is testing Faiss SQ for all supported memory-optimized-search widths
 * (bits ∈ {1, 2, 4}, corresponding to compression x32, x16, x8).
 */
public class MOSFaissSQIndexIT extends AbstractMemoryOptimizedKnnSearchIT {
    // These tests validate MOS off-heap behavior which is specific to the SQ code path.
    private static final String SQ_ENCODER_PARAMS = """
        {"encoder": {"name": "sq", "parameters": {"bits": 1}}}""";
    private static final String SQ_TWO_BIT_ENCODER_PARAMS = """
        {"encoder": {"name": "sq", "parameters": {"bits": 2}}}""";
    private static final String SQ_FOUR_BIT_ENCODER_PARAMS = """
        {"encoder": {"name": "sq", "parameters": {"bits": 4}}}""";

    @ExpectRemoteBuildValidation
    public void testNonNestedDiskBasedIndexWithIP() {
        // ANN search
        doTestNonNestedIndex(
            VectorDataType.FLOAT,
            SQ_ENCODER_PARAMS,
            false,
            SpaceType.INNER_PRODUCT,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x32
        );

        // Radial search
        doTestNonNestedIndex(
            VectorDataType.FLOAT,
            SQ_ENCODER_PARAMS,
            true,
            SpaceType.INNER_PRODUCT,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x32
        );
    }

    public void testNestedDiskBasedIndexWithIP() {
        // ANN search
        doTestNestedIndex(
            VectorDataType.FLOAT,
            SQ_ENCODER_PARAMS,
            SpaceType.INNER_PRODUCT,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x32
        );

        // We don't support radial search for nested index
    }

    @ExpectRemoteBuildValidation
    public void testNonNestedDiskBasedIndexWithL2() {
        // ANN search
        doTestNonNestedIndex(
            VectorDataType.FLOAT,
            SQ_ENCODER_PARAMS,
            false,
            SpaceType.L2,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x32
        );

        // Radial search
        doTestNonNestedIndex(
            VectorDataType.FLOAT,
            SQ_ENCODER_PARAMS,
            true,
            SpaceType.L2,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x32
        );
    }

    public void testNestedDiskBasedIndexWithL2() {
        // ANN search
        doTestNestedIndex(
            VectorDataType.FLOAT,
            SQ_ENCODER_PARAMS,
            SpaceType.L2,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x32
        );

        // We don't support radial search for nested index
    }

    @ExpectRemoteBuildValidation
    public void testNonNestedDiskBasedIndexWithCosine() {
        // ANN search
        doTestNonNestedIndex(
            VectorDataType.FLOAT,
            SQ_ENCODER_PARAMS,
            false,
            SpaceType.COSINESIMIL,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x32
        );

        // Radial search
        doTestNonNestedIndex(
            VectorDataType.FLOAT,
            SQ_ENCODER_PARAMS,
            true,
            SpaceType.COSINESIMIL,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x32
        );
    }

    public void testNestedDiskBasedIndexWithCosine() {
        // ANN search
        doTestNestedIndex(
            VectorDataType.FLOAT,
            SQ_ENCODER_PARAMS,
            SpaceType.COSINESIMIL,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x32
        );

        // We don't support radial search for nested index
    }

    // --- SQ 2-bit (x16 compression) ---

    @ExpectRemoteBuildValidation
    public void testNonNestedDiskBasedIndexWithIP_SQTwoBit() {
        doTestNonNestedIndex(
            VectorDataType.FLOAT,
            SQ_TWO_BIT_ENCODER_PARAMS,
            false,
            SpaceType.INNER_PRODUCT,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x16
        );
    }

    public void testNestedDiskBasedIndexWithIP_SQTwoBit() {
        doTestNestedIndex(
            VectorDataType.FLOAT,
            SQ_TWO_BIT_ENCODER_PARAMS,
            SpaceType.INNER_PRODUCT,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x16
        );
    }

    @ExpectRemoteBuildValidation
    public void testNonNestedDiskBasedIndexWithL2_SQTwoBit() {
        doTestNonNestedIndex(
            VectorDataType.FLOAT,
            SQ_TWO_BIT_ENCODER_PARAMS,
            false,
            SpaceType.L2,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x16
        );
    }

    public void testNestedDiskBasedIndexWithL2_SQTwoBit() {
        doTestNestedIndex(
            VectorDataType.FLOAT,
            SQ_TWO_BIT_ENCODER_PARAMS,
            SpaceType.L2,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x16
        );
    }

    @ExpectRemoteBuildValidation
    public void testNonNestedDiskBasedIndexWithCosine_SQTwoBit() {
        doTestNonNestedIndex(
            VectorDataType.FLOAT,
            SQ_TWO_BIT_ENCODER_PARAMS,
            false,
            SpaceType.COSINESIMIL,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x16
        );
    }

    public void testNestedDiskBasedIndexWithCosine_SQTwoBit() {
        doTestNestedIndex(
            VectorDataType.FLOAT,
            SQ_TWO_BIT_ENCODER_PARAMS,
            SpaceType.COSINESIMIL,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x16
        );
    }

    // --- SQ 4-bit (x8 compression) ---

    @ExpectRemoteBuildValidation
    public void testNonNestedDiskBasedIndexWithIP_SQFourBit() {
        doTestNonNestedIndex(
            VectorDataType.FLOAT,
            SQ_FOUR_BIT_ENCODER_PARAMS,
            false,
            SpaceType.INNER_PRODUCT,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x8
        );
    }

    public void testNestedDiskBasedIndexWithIP_SQFourBit() {
        doTestNestedIndex(
            VectorDataType.FLOAT,
            SQ_FOUR_BIT_ENCODER_PARAMS,
            SpaceType.INNER_PRODUCT,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x8
        );
    }

    @ExpectRemoteBuildValidation
    public void testNonNestedDiskBasedIndexWithL2_SQFourBit() {
        doTestNonNestedIndex(
            VectorDataType.FLOAT,
            SQ_FOUR_BIT_ENCODER_PARAMS,
            false,
            SpaceType.L2,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x8
        );
    }

    public void testNestedDiskBasedIndexWithL2_SQFourBit() {
        doTestNestedIndex(
            VectorDataType.FLOAT,
            SQ_FOUR_BIT_ENCODER_PARAMS,
            SpaceType.L2,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x8
        );
    }

    @ExpectRemoteBuildValidation
    public void testNonNestedDiskBasedIndexWithCosine_SQFourBit() {
        doTestNonNestedIndex(
            VectorDataType.FLOAT,
            SQ_FOUR_BIT_ENCODER_PARAMS,
            false,
            SpaceType.COSINESIMIL,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x8
        );
    }

    public void testNestedDiskBasedIndexWithCosine_SQFourBit() {
        doTestNestedIndex(
            VectorDataType.FLOAT,
            SQ_FOUR_BIT_ENCODER_PARAMS,
            SpaceType.COSINESIMIL,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x8
        );
    }
}

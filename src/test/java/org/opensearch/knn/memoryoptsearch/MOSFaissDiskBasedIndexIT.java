/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

/**
 * This is testing disk modes (32x, 16x, 8x) with LuceneOnFaiss is on.
 * Note that Faiss does not support 4x, and Faiss uses FP16 as 2x which is already covered in {@link MOSFaissFP16IndexIT}.
 */
public class MOSFaissDiskBasedIndexIT extends AbstractMemoryOptimizedKnnSearchIT {
    // Explicit BQ encoder params to pin x32 / x16 / x8 tests to the binary quantizer (not SQ).
    // Each compression tier auto-resolves to SQ post-3.9 default flip; these explicit encoder
    // params preserve the BQ code path coverage that used to run via EMPTY_PARAMS pre-flip.
    private static final String BQ_ENCODER_PARAMS_BITS_1 = """
        {"encoder": {"name": "binary", "parameters": {"bits": 1}}}""";
    private static final String BQ_ENCODER_PARAMS_BITS_2 = """
        {"encoder": {"name": "binary", "parameters": {"bits": 2}}}""";
    private static final String BQ_ENCODER_PARAMS_BITS_4 = """
        {"encoder": {"name": "binary", "parameters": {"bits": 4}}}""";

    public void testNonNestedDiskBasedIndexWithIP() {
        // x8 — SQ 4-bit via default, plus explicit BQ 4-bit to keep the BQ path covered.
        doTestNonNestedIndex(
            VectorDataType.FLOAT,
            EMPTY_PARAMS,
            false,
            SpaceType.INNER_PRODUCT,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x8
        );
        doTestNonNestedIndex(
            VectorDataType.FLOAT,
            BQ_ENCODER_PARAMS_BITS_4,
            false,
            SpaceType.INNER_PRODUCT,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x8
        );
        // x16 — SQ 2-bit via default, plus explicit BQ 2-bit.
        doTestNonNestedIndex(
            VectorDataType.FLOAT,
            EMPTY_PARAMS,
            false,
            SpaceType.INNER_PRODUCT,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x16
        );
        doTestNonNestedIndex(
            VectorDataType.FLOAT,
            BQ_ENCODER_PARAMS_BITS_2,
            false,
            SpaceType.INNER_PRODUCT,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x16
        );
        // x32 — explicit BQ 1-bit (default is SQ 1-bit; empty-params SQ path is already covered by MOSFaissSQIndexIT).
        doTestNonNestedIndex(
            VectorDataType.FLOAT,
            BQ_ENCODER_PARAMS_BITS_1,
            false,
            SpaceType.INNER_PRODUCT,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x32
        );
    }

    public void testNestedDiskBasedIndexWithIP() {
        doTestNestedIndex(
            VectorDataType.FLOAT,
            EMPTY_PARAMS,
            SpaceType.INNER_PRODUCT,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x8
        );
        doTestNestedIndex(
            VectorDataType.FLOAT,
            BQ_ENCODER_PARAMS_BITS_4,
            SpaceType.INNER_PRODUCT,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x8
        );
        doTestNestedIndex(
            VectorDataType.FLOAT,
            EMPTY_PARAMS,
            SpaceType.INNER_PRODUCT,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x16
        );
        doTestNestedIndex(
            VectorDataType.FLOAT,
            BQ_ENCODER_PARAMS_BITS_2,
            SpaceType.INNER_PRODUCT,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x16
        );
        doTestNestedIndex(
            VectorDataType.FLOAT,
            BQ_ENCODER_PARAMS_BITS_1,
            SpaceType.INNER_PRODUCT,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x32
        );
    }

    public void testWhenNoIndexBuiltForNonNested() {
        doTestNonNestedIndex(
            VectorDataType.FLOAT,
            EMPTY_PARAMS,
            false,
            SpaceType.INNER_PRODUCT,
            NO_BUILD_HNSW,
            Mode.ON_DISK,
            CompressionLevel.x8
        );
        doTestNonNestedIndex(
            VectorDataType.FLOAT,
            BQ_ENCODER_PARAMS_BITS_4,
            false,
            SpaceType.INNER_PRODUCT,
            NO_BUILD_HNSW,
            Mode.ON_DISK,
            CompressionLevel.x8
        );
        doTestNonNestedIndex(
            VectorDataType.FLOAT,
            EMPTY_PARAMS,
            false,
            SpaceType.INNER_PRODUCT,
            NO_BUILD_HNSW,
            Mode.ON_DISK,
            CompressionLevel.x16
        );
        doTestNonNestedIndex(
            VectorDataType.FLOAT,
            BQ_ENCODER_PARAMS_BITS_2,
            false,
            SpaceType.INNER_PRODUCT,
            NO_BUILD_HNSW,
            Mode.ON_DISK,
            CompressionLevel.x16
        );
        doTestNonNestedIndex(
            VectorDataType.FLOAT,
            BQ_ENCODER_PARAMS_BITS_1,
            false,
            SpaceType.INNER_PRODUCT,
            NO_BUILD_HNSW,
            Mode.ON_DISK,
            CompressionLevel.x32
        );
    }
}

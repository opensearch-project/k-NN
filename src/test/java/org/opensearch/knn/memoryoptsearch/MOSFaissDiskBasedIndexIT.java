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
 * This is testing disk modes (32x, 16x, 8x) with LuceneOnFaiss is on.
 * Note that Faiss does not support 4x, and Faiss uses FP16 as 2x which is already covered in {@link MOSFaissFP16IndexIT}.
 */
public class MOSFaissDiskBasedIndexIT extends AbstractMemoryOptimizedKnnSearchIT {
    @ExpectRemoteBuildValidation
    public void testNonNestedDiskBasedIndexWithIP() {
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
            EMPTY_PARAMS,
            false,
            SpaceType.INNER_PRODUCT,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x16
        );
        doTestNonNestedIndex(
            VectorDataType.FLOAT,
            EMPTY_PARAMS,
            false,
            SpaceType.INNER_PRODUCT,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x32
        );
    }

    @ExpectRemoteBuildValidation
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
            EMPTY_PARAMS,
            SpaceType.INNER_PRODUCT,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x16
        );
        doTestNestedIndex(
            VectorDataType.FLOAT,
            EMPTY_PARAMS,
            SpaceType.INNER_PRODUCT,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x32
        );
    }

    @ExpectRemoteBuildValidation
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
            EMPTY_PARAMS,
            false,
            SpaceType.INNER_PRODUCT,
            NO_BUILD_HNSW,
            Mode.ON_DISK,
            CompressionLevel.x16
        );
        doTestNonNestedIndex(
            VectorDataType.FLOAT,
            EMPTY_PARAMS,
            false,
            SpaceType.INNER_PRODUCT,
            NO_BUILD_HNSW,
            Mode.ON_DISK,
            CompressionLevel.x32
        );
    }

    public void testWhenNoIndexBuiltForNested() {
        doTestNestedIndex(VectorDataType.FLOAT, EMPTY_PARAMS, SpaceType.INNER_PRODUCT, NO_BUILD_HNSW, Mode.ON_DISK, CompressionLevel.x8);
        doTestNestedIndex(VectorDataType.FLOAT, EMPTY_PARAMS, SpaceType.INNER_PRODUCT, NO_BUILD_HNSW, Mode.ON_DISK, CompressionLevel.x16);
        doTestNestedIndex(VectorDataType.FLOAT, EMPTY_PARAMS, SpaceType.INNER_PRODUCT, NO_BUILD_HNSW, Mode.ON_DISK, CompressionLevel.x32);
    }
}

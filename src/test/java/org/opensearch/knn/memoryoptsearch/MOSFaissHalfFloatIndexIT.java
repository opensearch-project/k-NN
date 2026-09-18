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
 * Faiss HNSW coverage for the {@code half_float} data type across both supported compression levels:
 * x1 and x16 (SQ 1-bit).
 *
 * <p>Every test carries {@link ExpectRemoteBuildValidation} so the remote (GPU) build job asserts the index
 * was actually built remotely rather than silently falling back to a local build. Remote build eligibility
 * does not depend on nesting, so the nested cases are validated the same way.
 */
public class MOSFaissHalfFloatIndexIT extends AbstractMemoryOptimizedKnnSearchIT {

    // ---------------------------------------------------------------------
    // x1 - raw fp16 storage
    // ---------------------------------------------------------------------

    @ExpectRemoteBuildValidation
    public void testNonNestedHalfFloatIndexWithL2() {
        doTestNonNestedIndex(VectorDataType.HALF_FLOAT, EMPTY_PARAMS, false, SpaceType.L2, NO_ADDITIONAL_SETTINGS);
    }

    @ExpectRemoteBuildValidation
    public void testNonNestedHalfFloatIndexWithIP() {
        doTestNonNestedIndex(VectorDataType.HALF_FLOAT, EMPTY_PARAMS, false, SpaceType.INNER_PRODUCT, NO_ADDITIONAL_SETTINGS);
    }

    @ExpectRemoteBuildValidation
    public void testNonNestedHalfFloatIndexWithCosine() {
        doTestNonNestedIndex(VectorDataType.HALF_FLOAT, EMPTY_PARAMS, false, SpaceType.COSINESIMIL, NO_ADDITIONAL_SETTINGS);
    }

    @ExpectRemoteBuildValidation
    public void testNestedHalfFloatIndexWithL2() {
        doTestNestedIndex(VectorDataType.HALF_FLOAT, EMPTY_PARAMS, SpaceType.L2, NO_ADDITIONAL_SETTINGS);
    }

    @ExpectRemoteBuildValidation
    public void testNestedHalfFloatIndexWithIP() {
        doTestNestedIndex(VectorDataType.HALF_FLOAT, EMPTY_PARAMS, SpaceType.INNER_PRODUCT, NO_ADDITIONAL_SETTINGS);
    }

    @ExpectRemoteBuildValidation
    public void testNestedHalfFloatIndexWithCosine() {
        doTestNestedIndex(VectorDataType.HALF_FLOAT, EMPTY_PARAMS, SpaceType.COSINESIMIL, NO_ADDITIONAL_SETTINGS);
    }

    // ---------------------------------------------------------------------
    // x16 - SQ 1-bit
    // ---------------------------------------------------------------------

    @ExpectRemoteBuildValidation
    public void testNonNestedDiskBasedIndexWithL2_SQOneBit() {
        doTestNonNestedIndex(
            VectorDataType.HALF_FLOAT,
            EMPTY_PARAMS,
            false,
            SpaceType.L2,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x16
        );
    }

    @ExpectRemoteBuildValidation
    public void testNonNestedDiskBasedIndexWithIP_SQOneBit() {
        doTestNonNestedIndex(
            VectorDataType.HALF_FLOAT,
            EMPTY_PARAMS,
            false,
            SpaceType.INNER_PRODUCT,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x16
        );
    }

    @ExpectRemoteBuildValidation
    public void testNonNestedDiskBasedIndexWithCosine_SQOneBit() {
        doTestNonNestedIndex(
            VectorDataType.HALF_FLOAT,
            EMPTY_PARAMS,
            false,
            SpaceType.COSINESIMIL,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x16
        );
    }

    @ExpectRemoteBuildValidation
    public void testNestedDiskBasedIndexWithL2_SQOneBit() {
        doTestNestedIndex(
            VectorDataType.HALF_FLOAT,
            EMPTY_PARAMS,
            SpaceType.L2,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x16
        );
    }

    @ExpectRemoteBuildValidation
    public void testNestedDiskBasedIndexWithIP_SQOneBit() {
        doTestNestedIndex(
            VectorDataType.HALF_FLOAT,
            EMPTY_PARAMS,
            SpaceType.INNER_PRODUCT,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x16
        );
    }

    @ExpectRemoteBuildValidation
    public void testNestedDiskBasedIndexWithCosine_SQOneBit() {
        doTestNestedIndex(
            VectorDataType.HALF_FLOAT,
            EMPTY_PARAMS,
            SpaceType.COSINESIMIL,
            NO_ADDITIONAL_SETTINGS,
            Mode.ON_DISK,
            CompressionLevel.x16
        );
    }
}

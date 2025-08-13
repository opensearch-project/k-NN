/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.opensearch.knn.index.VectorDataType;

/**
 * Factory to get the right implementation of vector transfer
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class OffHeapVectorTransferFactory {

    /**
     * Gets the right vector transfer object based on vector data type
     * @param vectorDataType {@link VectorDataType}
     * @param bytesPerVector Bytes used per vector
     * @param totalVectorsToTransfer total number of vectors that will be transferred off heap
     * @return Correct implementation of {@link OffHeapVectorTransfer}
     * @param <T> float[] or byte[]
     */
    public static <T> OffHeapVectorTransfer<T> getVectorTransfer(
        final VectorDataType vectorDataType,
        int bytesPerVector,
        int totalVectorsToTransfer
    ) {
        switch (vectorDataType) {
            case FLOAT:
            case HALF_FLOAT:
                return (OffHeapVectorTransfer<T>) new OffHeapFloatVectorTransfer(bytesPerVector, totalVectorsToTransfer);
            case BINARY:
                return (OffHeapVectorTransfer<T>) new OffHeapBinaryVectorTransfer(bytesPerVector, totalVectorsToTransfer);
            case BYTE:
                return (OffHeapVectorTransfer<T>) new OffHeapByteVectorTransfer(bytesPerVector, totalVectorsToTransfer);
            default:
                throw new IllegalArgumentException("Unsupported vector data type: " + vectorDataType);
        }
    }
}

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
     * @param transferLimit max number of vectors that can be transferred to off heap in one transfer
     * @return Correct implementation of {@link OffHeapVectorTransfer}
     * @param <T> float[] or byte[]
     */
    public static <T> OffHeapVectorTransfer<T> getVectorTransfer(final VectorDataType vectorDataType, final int transferLimit) {
        switch (vectorDataType) {
            case FLOAT:
                return (OffHeapVectorTransfer<T>) new OffHeapFloatVectorTransfer(transferLimit);
            case BINARY:
                return (OffHeapVectorTransfer<T>) new OffHeapBinaryVectorTransfer(transferLimit);
            case BYTE:
                return (OffHeapVectorTransfer<T>) new OffHeapByteVectorTransfer(transferLimit);
            default:
                throw new IllegalArgumentException("Unsupported vector data type: " + vectorDataType);
        }
    }
}

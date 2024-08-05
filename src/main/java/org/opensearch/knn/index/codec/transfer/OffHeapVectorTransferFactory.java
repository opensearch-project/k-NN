/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer;

import org.opensearch.knn.index.VectorDataType;

public final class OffHeapVectorTransferFactory {

    private OffHeapVectorTransferFactory() {}

    public static <T> OffHeapVectorTransfer<T> getVectorTransfer(final VectorDataType vectorDataType, final int transferLimit) {
        switch (vectorDataType) {
            case FLOAT:
                return (OffHeapVectorTransfer<T>) new OffHeapFloatVectorTransfer(transferLimit);
            case BINARY:
                // TODO: Add binary here
            case BYTE:
                return (OffHeapVectorTransfer<T>) new OffHeapByteVectorTransfer(transferLimit);
            default:
                throw new IllegalArgumentException("Unsupported vector data type: " + vectorDataType);
        }
    }
}

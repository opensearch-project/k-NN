/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.Getter;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;

public class MMapByteVectorValues extends ByteVectorValues {
    private final IndexInput indexInput;
    @Getter
    private final long oneVectorByteSize;
    @Getter
    private final long baseOffset;
    private final int dimension;
    private final int totalNumberOfVectors;
    // It has address and size per MemorySegment extracted from MemorySegmentIndexInput.
    // e.g. address_i = addressAndSize[i], mapped_size_i = addressAndSize[i + 1]
    @Getter
    private final long[] addressAndSize;
    private byte[] buffer;

    public MMapByteVectorValues(
        final IndexInput indexInput,
        final long oneVectorByteSize,
        final long baseOffset,
        final int dimension,
        final int totalNumberOfVectors,
        final long[] addressAndSize
    ) {
        this.indexInput = indexInput;
        this.oneVectorByteSize = oneVectorByteSize;
        this.baseOffset = baseOffset;
        this.dimension = dimension;
        this.totalNumberOfVectors = totalNumberOfVectors;
        if (addressAndSize == null || addressAndSize.length == 0) {
            throw new IllegalArgumentException("Empty `addressAndSize` was provided in " + MMapByteVectorValues.class.getSimpleName());
        }
        this.addressAndSize = addressAndSize;
    }

    @Override
    public byte[] vectorValue(int internalVectorId) throws IOException {
        indexInput.seek(baseOffset + internalVectorId * oneVectorByteSize);
        // Lazy initialization, in general this method is not expected to be called during search.
        // During search, distance calculation will be done in a separated C++ code.
        if (buffer == null) {
            buffer = new byte[(int) oneVectorByteSize];
        }
        indexInput.readBytes(buffer, 0, buffer.length);
        return buffer;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public int size() {
        return totalNumberOfVectors;
    }

    @Override
    public ByteVectorValues copy() throws IOException {
        return new MMapByteVectorValues(indexInput.clone(), oneVectorByteSize, baseOffset, dimension, totalNumberOfVectors, addressAndSize);
    }
}

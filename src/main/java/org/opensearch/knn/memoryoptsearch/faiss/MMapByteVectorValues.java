/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.Getter;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;

/**
 * Encapsulates raw pointers to memory-mapped regions associated with a given {@link IndexInput}.
 * <p>
 * The list of addresses and sizes can be passed to native scoring code for improved performance.
 * Native code may operate directly on these mapped pointers to perform computations efficiently.
 * <p>
 * Users can still retrieve vector bytes via the {@code vectorValue} API, which lazily creates
 * an internal buffer and returns it after filling in the requested bytes.
 */
public class MMapByteVectorValues extends ByteVectorValues implements MMapVectorValues {
    private final IndexInput indexInput;
    // oneVectorByteSize == Float.BYTES * Dimension. Ex: 3072 bytes for 768 dimensions.
    private final long oneVectorByteSize;
    // Start offset pointing to flat vectors section in Faiss index.
    private final long baseOffset;
    // Vector dimension
    private final int dimension;
    // Total number of vectors stored in Faiss index.
    private final int totalNumberOfVectors;
    // It has address and size per MemorySegment extracted from MemorySegmentIndexInput.
    // e.g. address_i = addressAndSize[i], mapped_size_i = addressAndSize[i + 1]
    // For example, if a given IndexInput had 2 MemorySegments, then addressAndSize[0] has the address of the first MemorySegment
    // and addressAndSize[1] has the size of mapped region. Similarly, addressAndSize[2] has the address of the second MemorySegment, and
    // addressAndSize[3] has the size of the second mapped region.
    @Getter
    private final long[] addressAndSize;
    // Internal buffer that lazily created.
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
            throw new IllegalArgumentException(
                "Empty `addressAndSize` was provided in " + MMapByteVectorValues.class.getSimpleName() + ". Is null?=" + (addressAndSize
                                                                                                                          == null));
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

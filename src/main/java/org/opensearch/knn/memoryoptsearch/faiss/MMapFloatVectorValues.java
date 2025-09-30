/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.Getter;
import org.apache.lucene.index.FloatVectorValues;
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
public class MMapFloatVectorValues extends FloatVectorValues {
    private final IndexInput indexInput;
    @Getter
    // oneVectorByteSize == Float.BYTES * Dimension. Ex: 3072 bytes for 768 dimensions.
    private final long oneVectorByteSize;
    @Getter
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
    private float[] buffer;

    public MMapFloatVectorValues(
        final IndexInput indexInput,
        final long baseOffset,
        final int dimension,
        final int totalNumberOfVectors,
        final long[] addressAndSize
    ) {
        this.indexInput = indexInput;
        this.oneVectorByteSize = Float.BYTES * dimension;
        this.baseOffset = baseOffset;
        this.dimension = dimension;
        this.totalNumberOfVectors = totalNumberOfVectors;
        if (addressAndSize == null || addressAndSize.length == 0) {
            throw new IllegalArgumentException(
                "Empty `addressAndSize` was provided in "
                    + MMapFloatVectorValues.class.getSimpleName()
                    + ". Is null?="
                    + (addressAndSize == null)
            );
        }
        this.addressAndSize = addressAndSize;
    }

    @Override
    public float[] vectorValue(int internalVectorId) throws IOException {
        indexInput.seek(baseOffset + internalVectorId * oneVectorByteSize);
        // Lazy initialization, in general this method is not expected to be called during search.
        // During search, distance calculation will be done in a separated C++ code.
        if (buffer == null) {
            buffer = new float[(int) dimension];
        }
        indexInput.readFloats(buffer, 0, buffer.length);
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
    public FloatVectorValues copy() throws IOException {
        return new MMapFloatVectorValues(indexInput.clone(), baseOffset, dimension, totalNumberOfVectors, addressAndSize);
    }
}

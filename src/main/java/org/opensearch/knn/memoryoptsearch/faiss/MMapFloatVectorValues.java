/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.Getter;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;

/**
 * A {@link FloatVectorValues} wrapper that exposes memory-mapped segment addresses for direct native access.
 * <p>
 * This class decorates a delegate {@link FloatVectorValues} backed by an MMap {@link IndexInput}, exposing
 * the underlying memory segment addresses and sizes via {@link #getAddressAndSize()}. Native scoring code
 * can use these raw pointers to read vectors directly from mapped memory, avoiding Java-level copies.
 * <p>
 * Standard Java access is still available through {@link #vectorValue(int)}, which delegates to the
 * underlying implementation. The class also implements {@link HasIndexSlice} to provide access to the
 * backing {@link IndexInput} for prefetching or sequential reads.
 */
public class MMapFloatVectorValues extends FloatVectorValues implements MMapVectorValues, HasIndexSlice {
    // It has address and size per MemorySegment extracted from MemorySegmentIndexInput.
    // e.g. address_i = addressAndSize[i], mapped_size_i = addressAndSize[i + 1]
    // For example, if a given IndexInput had 2 MemorySegments, then addressAndSize[0] has the address of the first MemorySegment
    // and addressAndSize[1] has the size of mapped region. Similarly, addressAndSize[2] has the address of the second MemorySegment, and
    // addressAndSize[3] has the size of the second mapped region.
    @Getter
    private final long[] addressAndSize;
    private final FloatVectorValues delegate;

    public MMapFloatVectorValues(final FloatVectorValues delegate, final long[] addressAndSize) {
        this.delegate = delegate;
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
        return delegate.vectorValue(internalVectorId);
    }

    @Override
    public int dimension() {
        return delegate.dimension();
    }

    @Override
    public int size() {
        return delegate.size();
    }

    /**
     * Returns the vector byte length, defaults to dimension multiplied by float byte size. Hence, we are changing it to
     * oneVectorByteSize
     */
    @Override
    public int getVectorByteLength() {
        return delegate.getVectorByteLength();
    }

    @Override
    public FloatVectorValues copy() throws IOException {
        return new MMapFloatVectorValues(delegate.copy(), addressAndSize);
    }

    /**
     * Returns an IndexInput from which to read this instance's values, or null if not available.
     */
    @Override
    public IndexInput getSlice() {
        if (delegate instanceof HasIndexSlice hasIndexSlice) {
            return hasIndexSlice.getSlice();
        }
        return null;
    }
}

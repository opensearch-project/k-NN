/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.vectorvalues;

import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;

/**
 * A {@link ByteVectorValues} implementation that reads byte-encoded vectors directly from a FAISS
 * index section via an {@link IndexInput}.
 * <p>
 * Each vector is located by seeking to {@code internalVectorId * codeSize} within the backing
 * {@link IndexInput} and reading {@code codeSize} bytes. This is used for scalar-quantized vectors
 * where {@code codeSize} may differ from {@code dimension} (e.g., SQ8 uses 1 byte per dimension,
 * while SQfp16 uses 2 bytes per dimension).
 * <p>
 * A single reusable buffer is used across calls to {@link #vectorValue(int)}, so callers must
 * consume or copy the returned array before the next call.
 * <p>
 * Implements {@link HasIndexSlice} to expose the underlying {@link IndexInput} for prefetching
 * or direct access by native code.
 */
public class FaissByteVectorValues extends ByteVectorValues implements HasIndexSlice {
    private final IndexInput indexInput;
    private final byte[] buffer;
    private final int codeSize;
    private final int dimension;
    private final int totalNumberOfVectors;

    /**
     * @param indexInput           The {@link IndexInput} positioned at the start of the vector data section.
     * @param codeSize             The byte size of a single encoded vector (may differ from dimension for quantized vectors).
     * @param dimension            The logical vector dimension.
     * @param totalNumberOfVectors The total number of vectors in this section.
     */
    public FaissByteVectorValues(final IndexInput indexInput, int codeSize, int dimension, int totalNumberOfVectors) {
        this.indexInput = indexInput;
        this.codeSize = codeSize;
        this.dimension = dimension;
        this.totalNumberOfVectors = totalNumberOfVectors;
        this.buffer = new byte[codeSize];
    }

    /**
     * Return the vector value for the given vector ordinal which must be in [0, size() - 1],
     * otherwise IndexOutOfBoundsException is thrown. The returned array may be shared across calls.
     *
     * @return the vector value
     */
    @Override
    public byte[] vectorValue(int internalVectorId) throws IOException {
        final long offset = (long) internalVectorId * codeSize;
        indexInput.seek(offset);
        indexInput.readBytes(buffer, 0, codeSize);
        return buffer;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    /**
     * Returns the byte length of a single vector, which equals {@code codeSize}. Default lucene returns
     * dimension multiplied by float byte size, hence we need to override this method
     */
    @Override
    public int getVectorByteLength() {
        return codeSize;
    }

    @Override
    public int size() {
        return totalNumberOfVectors;
    }

    @Override
    public ByteVectorValues copy() {
        return new FaissByteVectorValues(indexInput.clone(), codeSize, dimension, totalNumberOfVectors);
    }

    /**
     * Returns an IndexInput from which to read this instance's values, or null if not available.
     */
    @Override
    public IndexInput getSlice() {
        return indexInput;
    }
}

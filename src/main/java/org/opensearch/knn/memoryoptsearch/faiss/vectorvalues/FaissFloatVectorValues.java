/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.vectorvalues;

import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;

/**
 * A {@link FloatVectorValues} implementation that reads float vectors directly from a FAISS index
 * section via an {@link IndexInput}.
 * <p>
 * Each vector is located by seeking to {@code internalVectorId * oneVectorByteSize} within the
 * backing {@link IndexInput} and reading {@code dimension} floats. A single reusable buffer is
 * used across calls to {@link #vectorValue(int)}, so callers must consume or copy the returned
 * array before the next call.
 * <p>
 * Implements {@link HasIndexSlice} to expose the underlying {@link IndexInput} for prefetching
 * or direct access by native code.
 */
public class FaissFloatVectorValues extends FloatVectorValues implements HasIndexSlice {
    private final IndexInput indexInput;
    private final long oneVectorByteSize;
    private final int dimension;
    private final float[] buffer;
    private final int totalNumberOfVectors;

    public FaissFloatVectorValues(IndexInput indexInput, long oneVectorByteSize, int dimension, int totalNumberOfVectors) {
        this.indexInput = indexInput;
        this.oneVectorByteSize = oneVectorByteSize;
        this.dimension = dimension;
        this.buffer = new float[dimension];
        this.totalNumberOfVectors = totalNumberOfVectors;
    }

    @Override
    public float[] vectorValue(int internalVectorId) throws IOException {
        indexInput.seek(internalVectorId * oneVectorByteSize);
        indexInput.readFloats(buffer, 0, buffer.length);
        return buffer;
    }

    @Override
    public FloatVectorValues copy() {
        return new FaissFloatVectorValues(indexInput.clone(), oneVectorByteSize, dimension, totalNumberOfVectors);
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public int size() {
        return totalNumberOfVectors;
    }

    /**
     * Returns an IndexInput from which to read this instance's values, or null if not available.
     */
    @Override
    public IndexInput getSlice() {
        return indexInput;
    }
}

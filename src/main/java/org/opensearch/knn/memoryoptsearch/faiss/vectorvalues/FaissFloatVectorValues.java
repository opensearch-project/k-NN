/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.vectorvalues;

import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.packed.DirectMonotonicReader;
import org.opensearch.knn.memoryoptsearch.faiss.WrappedFloatVectorValues;

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

    /**
     * A {@link FloatVectorValues} wrapper for sparse or nested cases that maps internal vector IDs
     * to Lucene document IDs via a {@link DirectMonotonicReader}.
     * <p>
     * Delegates vector reads to the wrapped {@link FloatVectorValues} and translates ordinals
     * in {@link #ordToDoc(int)} and {@link #getAcceptOrds(Bits)}.
     */
    public static class SparseFloatVectorValuesImpl extends WrappedFloatVectorValues implements HasIndexSlice {
        private final DirectMonotonicReader idMappingReader;

        public SparseFloatVectorValuesImpl(final FloatVectorValues vectorValues, final DirectMonotonicReader idMappingReader) {
            super(vectorValues);
            this.idMappingReader = idMappingReader;
            if ((vectorValues instanceof HasIndexSlice) == false) {
                throw new IllegalArgumentException(
                    "SparseFloatVectorValuesImpl needs an instance of "
                        + "FloatVectorValues which implements HasIndexSlice interface. "
                        + vectorValues.getClass().getCanonicalName()
                        + " doesn't implement "
                        + HasIndexSlice.class
                        + "interface"
                );
            }
        }

        @Override
        public float[] vectorValue(int internalVectorId) throws IOException {
            return floatVectorValues.vectorValue(internalVectorId);
        }

        @Override
        public int dimension() {
            return floatVectorValues.dimension();
        }

        @Override
        public int ordToDoc(int internalVectorId) {
            return (int) idMappingReader.get(internalVectorId);
        }

        @Override
        public int getVectorByteLength() {
            return floatVectorValues.getVectorByteLength();
        }

        @Override
        public Bits getAcceptOrds(final Bits acceptDocs) {
            if (acceptDocs != null) {
                return new Bits() {
                    @Override
                    public boolean get(int internalVectorId) {
                        return acceptDocs.get((int) idMappingReader.get(internalVectorId));
                    }

                    @Override
                    public int length() {
                        return floatVectorValues.size();
                    }
                };
            }
            return null;
        }

        @Override
        public int size() {
            return floatVectorValues.size();
        }

        @Override
        public FloatVectorValues copy() throws IOException {
            return new SparseFloatVectorValuesImpl(floatVectorValues.copy(), idMappingReader);
        }

        @Override
        public IndexInput getSlice() {
            // Since in constructor we are already validating the instance type we don't need another validation here
            return ((HasIndexSlice) floatVectorValues).getSlice();
        }
    }
}

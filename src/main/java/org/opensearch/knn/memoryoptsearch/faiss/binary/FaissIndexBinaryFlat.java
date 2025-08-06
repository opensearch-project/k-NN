/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.binary;

import lombok.RequiredArgsConstructor;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.memoryoptsearch.FlatVectorsReaderWithFieldName;
import org.opensearch.knn.memoryoptsearch.faiss.FaissSection;

import java.io.IOException;

/**
 * Binary flat vector format for a Faiss index.
 * <p>
 * The format consists of two parts:
 * 1. A binary header
 * 2. A flat binary vector section
 * Note: Binary vectors stored within this format should be compared using Hamming distance only.
 * See <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/IndexBinaryFlat.h">...</a> for more details.
 */
public class FaissIndexBinaryFlat extends FaissBinaryIndex {
    public static final String IBXF = "IBxF";

    private FaissSection binaryFlatVectorSection;

    public FaissIndexBinaryFlat() {
        super(IBXF);
    }

    /**
     * Partial load the flat float vector section which is code_size * total_number_of_vectors.
     * FYI FAISS - <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/impl/index_read.cpp#L1363">...</a>
     *
     * @param input Index input reading bytes from Faiss index.
     * @throws IOException
     */
    @Override
    protected void doLoad(IndexInput input, FlatVectorsReaderWithFieldName flatVectorsReaderWithFieldName) throws IOException {
        readBinaryCommonHeader(input);
        binaryFlatVectorSection = new FaissSection(input, 1);
    }

    @Override
    public VectorEncoding getVectorEncoding() {
        return VectorEncoding.BYTE;
    }

    @Override
    public FloatVectorValues getFloatValues(IndexInput indexInput) throws IOException {
        throw new UnsupportedOperationException(FaissIndexBinaryFlat.class.getSimpleName() + " does not support FloatValues.");
    }

    @Override
    public ByteVectorValues getByteValues(final IndexInput indexInput) throws IOException {
        @RequiredArgsConstructor
        class ByteVectorValuesImpl extends ByteVectorValues {
            final IndexInput indexInput;
            final byte[] buffer = new byte[codeSize];

            @Override
            public byte[] vectorValue(int internalVectorId) throws IOException {
                final long offset = binaryFlatVectorSection.getBaseOffset() + (long) internalVectorId * codeSize;
                indexInput.seek(offset);
                indexInput.readBytes(buffer, 0, codeSize);
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
            public ByteVectorValues copy() {
                return new ByteVectorValuesImpl(indexInput.clone());
            }
        }

        return new ByteVectorValuesImpl(indexInput);
    }
}

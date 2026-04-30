/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.Getter;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissBinaryIndex;

import java.io.IOException;

/**
 * A virtual FaissIndex that serves as a proxy for Lucene's BinaryQuantized vectors reader.
 * When Faiss SQ (for 1 bit) is used, the HNSW graph is stored in the .faiss file without storage (IO_FLAG_SKIP_STORAGE),
 * and quantized vectors are stored separately via Lucene's format. This class bridges the two by installing
 * itself as the flat storage under FaissHNSWIndex, providing access to the quantized vectors reader for scoring.
 */
public class FaissScalarQuantizedFlatIndex extends FaissBinaryIndex {
    static final String FAISS_SCALAR_QUANTIZED_FLAT_INDEX = "FaissScalarQuantizedFlatIndex";

    @Getter
    private final FlatVectorsReader flatVectorsReader;
    @Getter
    private final String fieldName;

    public FaissScalarQuantizedFlatIndex(final FlatVectorsReader flatVectorsReader, final String fieldName) {
        super(FAISS_SCALAR_QUANTIZED_FLAT_INDEX);
        this.flatVectorsReader = flatVectorsReader;
        this.fieldName = fieldName;
    }

    @Override
    protected void doLoad(IndexInput input) throws IOException {
        // No-op: quantized vectors are managed by LuceneSQFlatReader, not loaded from the faiss file.
    }

    @Override
    public VectorEncoding getVectorEncoding() {
        return VectorEncoding.FLOAT32;
    }

    /**
     * Returns a {@code ScalarQuantizedVectorValues} containing both raw and quantized vector values.
     * Raw vectors are stored in Lucene's segment files and provide full-precision floats via
     * {@code vectorValue(ord)}, while quantized vectors are used for scoring via {@code scorer()}.
     */
    @Override
    public FloatVectorValues getFloatValues(IndexInput indexInput) throws IOException {
        return flatVectorsReader.getFloatVectorValues(fieldName);
    }

    @Override
    public ByteVectorValues getByteValues(IndexInput indexInput) throws IOException {
        throw new UnsupportedOperationException(
            String.format("%s does not support byte vector values.", FAISS_SCALAR_QUANTIZED_FLAT_INDEX)
        );
    }
}

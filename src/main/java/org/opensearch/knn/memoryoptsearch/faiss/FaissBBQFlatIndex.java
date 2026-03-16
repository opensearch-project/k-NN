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

import java.io.IOException;

/**
 * A virtual FaissIndex that serves as a proxy for Lucene's BinaryQuantized vectors reader.
 * When Faiss BBQ is used, the HNSW graph is stored in the .faiss file without storage (IO_FLAG_SKIP_STORAGE),
 * and quantized vectors are stored separately via Lucene's format. This class bridges the two by installing
 * itself as the flat storage under FaissHNSWIndex, providing access to the quantized vectors reader for scoring.
 */
public class FaissBBQFlatIndex extends FaissIndex {
    static final String FAISS_BBQ_FLAT_INDEX = "FaissBBQFlatIndex";

    @Getter
    private final FlatVectorsReader bbqFlatReader;
    @Getter
    private final String fieldName;

    public FaissBBQFlatIndex(final FlatVectorsReader bbqFlatReader, final String fieldName) {
        super(FAISS_BBQ_FLAT_INDEX);
        this.bbqFlatReader = bbqFlatReader;
        this.fieldName = fieldName;
    }

    @Override
    protected void doLoad(IndexInput input) throws IOException {
        // No-op: quantized vectors are managed by bbqFlatReader, not loaded from the faiss file.
    }

    @Override
    public VectorEncoding getVectorEncoding() {
        return VectorEncoding.FLOAT32;
    }

    @Override
    public FloatVectorValues getFloatValues(IndexInput indexInput) throws IOException {
        return bbqFlatReader.getFloatVectorValues(fieldName);
    }

    @Override
    public ByteVectorValues getByteValues(IndexInput indexInput) throws IOException {
        throw new UnsupportedOperationException("FaissBBQFlatIndex does not support byte vector values.");
    }
}

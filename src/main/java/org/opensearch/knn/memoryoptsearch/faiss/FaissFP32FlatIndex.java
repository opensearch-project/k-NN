/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import java.util.Locale;

import lombok.Getter;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;

/**
 * A virtual {@link FaissIndex} that serves full-precision FP32 vectors from Lucene's {@link FlatVectorsReader} instead
 * of from the .faiss file. When flat-vector deduplication is enabled ({@code index.knn.advanced.flat_vector_dedup}), a
 * FAISS FP32 HNSW graph is written without its embedded flat storage (IO_FLAG_SKIP_STORAGE), and the full-precision
 * vectors live only in Lucene's .vec file. This class installs itself as the flat storage under {@link FaissHNSWIndex},
 * delegating vector access to the Lucene reader.
 *
 * <p>Mirrors {@link FaissScalarQuantizedFlatIndex}, but extends {@link FaissIndex} directly (float storage is set via
 * {@link AbstractFaissHNSWIndex#setFlatVectors}, which accepts any {@link FaissIndex}) and serves non-quantized floats.
 */
public class FaissFP32FlatIndex extends FaissIndex {
    static final String FAISS_FP32_FLAT_INDEX = "FaissFP32FlatIndex";

    @Getter
    private final FlatVectorsReader flatVectorsReader;
    @Getter
    private final String fieldName;

    public FaissFP32FlatIndex(final FlatVectorsReader flatVectorsReader, final String fieldName) {
        super(FAISS_FP32_FLAT_INDEX);
        this.flatVectorsReader = flatVectorsReader;
        this.fieldName = fieldName;
    }

    @Override
    protected void doLoad(IndexInput input) throws IOException {
        // No-op: full-precision vectors are managed by Lucene's FlatVectorsReader, not loaded from the .faiss file.
    }

    @Override
    public VectorEncoding getVectorEncoding() {
        return VectorEncoding.FLOAT32;
    }

    @Override
    public FloatVectorValues getFloatValues(IndexInput indexInput) throws IOException {
        return flatVectorsReader.getFloatVectorValues(fieldName);
    }

    @Override
    public ByteVectorValues getByteValues(IndexInput indexInput) throws IOException {
        throw new UnsupportedOperationException(
            String.format(Locale.ROOT, "%s does not support byte vector values.", FAISS_FP32_FLAT_INDEX)
        );
    }
}

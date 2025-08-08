/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.binary;

import lombok.Getter;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.memoryoptsearch.FlatVectorsReaderWithFieldName;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHNSW;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHNSWProvider;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndexLoadUtils;

import java.io.IOException;

/**
 * HNSW graph having a binary storage internally in Faiss index file.
 * <p>
 * Faiss - <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/IndexBinaryHNSW.h">...</a>
 */
public class FaissBinaryHnswIndex extends FaissBinaryIndex implements FaissHNSWProvider {
    public static final String IBHF = "IBHf";

    @Getter
    protected FaissHNSW faissHnsw;
    protected FaissBinaryIndex storage;

    public FaissBinaryHnswIndex(final String indexType, final FaissHNSW faissHnsw) {
        super(indexType);
        this.faissHnsw = faissHnsw;
    }

    /**
     * Partial load binary HNSW index via the provided {@link IndexInput}.
     * <p>
     * Faiss - <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/impl/index_read.cpp#L1381">...</a>
     *
     * @param input
     * @throws IOException
     */
    @Override
    protected void doLoad(IndexInput input, FlatVectorsReaderWithFieldName flatVectorsReaderWithFieldName) throws IOException {
        // Read common binary index header
        readBinaryCommonHeader(input);

        // Partial load HNSW graph
        faissHnsw.load(input, getTotalNumberOfVectors());

        // Partial load storage
        storage = FaissIndexLoadUtils.toBinaryIndex(FaissIndex.load(input, flatVectorsReaderWithFieldName));
    }

    @Override
    public VectorEncoding getVectorEncoding() {
        return storage.getVectorEncoding();
    }

    @Override
    public FloatVectorValues getFloatValues(IndexInput indexInput) throws IOException {
        throw new UnsupportedOperationException(FaissBinaryHnswIndex.class.getSimpleName() + " does not support FloatVectorValues.");
    }

    @Override
    public ByteVectorValues getByteValues(IndexInput indexInput) throws IOException {
        return storage.getByteValues(indexInput);
    }
}

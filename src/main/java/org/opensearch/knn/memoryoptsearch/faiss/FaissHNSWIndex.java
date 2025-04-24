/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;

/**
 * A flat HNSW index that contains both an HNSW graph and flat vector storage.
 * This is the ported version of `IndexHNSW` from FAISS.
 * For more details, please refer to <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/impl/index_read.cpp#L1111">...</a>
 */
public class FaissHNSWIndex extends AbstractFaissHNSWIndex {
    // Flat float vector format -
    // https://github.com/facebookresearch/faiss/blob/15491a1e4f5a513a8684e5b7262ef4ec22eda19d/faiss/IndexHNSW.h#L122
    public static final String IHNF = "IHNf";
    // Quantized flat format with HNSW -
    // https://github.com/facebookresearch/faiss/blob/15491a1e4f5a513a8684e5b7262ef4ec22eda19d/faiss/IndexHNSW.h#L144C8-L144C19
    public static final String IHNS = "IHNs";

    public FaissHNSWIndex(final String indexType) {
        super(indexType, new FaissHNSW());

        if (indexType.equals(IHNF) == false && indexType.equals(IHNS) == false) {
            throw new IllegalStateException("Unsupported index type: [" + indexType + "] in " + FaissHNSWIndex.class.getSimpleName());
        }
    }

    /**
     * Loading HNSW graph and nested storage index.
     * For more details, please refer to
     * <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/impl/index_read.cpp#L1111">...</a>
     * @param input
     * @throws IOException
     */
    @Override
    protected void doLoad(IndexInput input) throws IOException {
        // Read common header
        readCommonHeader(input);

        // Partial load HNSW graph
        faissHnsw.load(input, getTotalNumberOfVectors());

        // Partial load flat vector storage
        flatVectors = FaissIndex.load(input);
    }

    @Override
    public VectorEncoding getVectorEncoding() {
        return flatVectors.getVectorEncoding();
    }

    @Override
    public FloatVectorValues getFloatValues(IndexInput indexInput) throws IOException {
        return flatVectors.getFloatValues(indexInput);
    }

    @Override
    public ByteVectorValues getByteValues(IndexInput indexInput) throws IOException {
        return flatVectors.getByteValues(indexInput);
    }
}

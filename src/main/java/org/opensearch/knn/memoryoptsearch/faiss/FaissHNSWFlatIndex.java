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
 * For more details, please refer to <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/IndexHNSW.h">...</a>
 */
public class FaissHNSWFlatIndex extends FaissIndex {
    public FaissHNSWFlatIndex(final String indexType) {
        super(indexType);
    }

    @Override
    protected void doLoad(IndexInput input) throws IOException {
        // TODO(KDY) : This will be covered in part-3 (FAISS HNSW).
    }

    @Override
    public VectorEncoding getVectorEncoding() {
        // TODO(KDY) : This will be covered in part-3 (FAISS HNSW).
        return null;
    }

    @Override
    public FloatVectorValues getFloatValues(IndexInput indexInput) throws IOException {
        // TODO(KDY) : This will be covered in part-3 (FAISS HNSW).
        return null;
    }

    @Override
    public ByteVectorValues getByteValues(IndexInput indexInput) throws IOException {
        // TODO(KDY) : This will be covered in part-3 (FAISS HNSW).
        return null;
    }
}

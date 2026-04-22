/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexInput;

/**
 * Sentinel {@link FaissIndex} returned when a "null" section is encountered during loading
 * (e.g., Faiss SQ (for 1 bit) where flat storage is skipped via IO_FLAG_SKIP_STORAGE).
 * The caller is expected to replace this with a concrete flat index (e.g., {@link FaissScalarQuantizedFlatIndex}).
 */
public class FaissEmptyIndex extends FaissIndex {
    public static final FaissEmptyIndex INSTANCE = new FaissEmptyIndex();

    private FaissEmptyIndex() {
        super(NULL_INDEX_TYPE);
    }

    @Override
    protected void doLoad(IndexInput input) {}

    @Override
    public VectorEncoding getVectorEncoding() {
        throw new UnsupportedOperationException(String.format("%s does not support this operation.", getClass().getSimpleName()));
    }

    @Override
    public FloatVectorValues getFloatValues(IndexInput indexInput) {
        throw new UnsupportedOperationException(String.format("%s does not support this operation.", getClass().getSimpleName()));
    }

    @Override
    public ByteVectorValues getByteValues(IndexInput indexInput) {
        throw new UnsupportedOperationException(String.format("%s does not support this operation.", getClass().getSimpleName()));
    }

    /**
     * Checks whether the given {@link FaissIndex} is an empty placeholder written by Faiss
     * when {@code IO_FLAG_SKIP_STORAGE} is used (e.g., Faiss SQ 1-bit where flat vector
     * storage is omitted from the .faiss file and instead served by Lucene's flat files).
     *
     * @param maybeEmptyStorage the index loaded from the storage section of a Faiss HNSW file
     * @return {@code true} if the index is a {@link FaissEmptyIndex} placeholder
     */
    public static boolean isEmptyIndex(final FaissIndex maybeEmptyStorage) {
        return maybeEmptyStorage instanceof FaissEmptyIndex;
    }
}

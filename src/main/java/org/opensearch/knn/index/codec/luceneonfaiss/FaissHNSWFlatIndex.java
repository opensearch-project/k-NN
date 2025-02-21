/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.luceneonfaiss;

import lombok.Getter;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;

/**
 * A flat HNSW index that contains both an HNSW graph and flat vector storage.
 * This is the ported version of `IndexHNSW` from FAISS.
 * For more details, please refer to <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/IndexHNSW.h">...</a>
 */
public class FaissHNSWFlatIndex extends FaissIndex {
    public static final String IHNF = "IHNf";

    @Getter
    private FaissHNSW hnsw = new FaissHNSW();
    @Getter
    private FaissIndexFlat storage;

    /**
     * Partially loads both the HNSW graph and the underlying flat vectors.
     *
     * @param input An input stream for a FAISS HNSW graph file, allowing access to the neighbor list and vector locations.
     * @return {@link FaissHNSWFlatIndex} instance consists of index hierarchy.
     * @throws IOException
     */
    public static FaissHNSWFlatIndex load(IndexInput input) throws IOException {
        // Read common header
        FaissHNSWFlatIndex faissHNSWFlatIndex = new FaissHNSWFlatIndex();
        readCommonHeader(input, faissHNSWFlatIndex);

        // Partial load HNSW graph
        faissHNSWFlatIndex.hnsw = FaissHNSW.load(input, faissHNSWFlatIndex.getTotalNumberOfVectors());

        // Partial load flat vector storage
        final FaissIndex faissIndex = FaissIndex.load(input);
        if (faissIndex instanceof FaissIndexFlat) {
            faissHNSWFlatIndex.storage = (FaissIndexFlat) faissIndex;
        } else {
            throw new IllegalStateException(
                "Expected flat vector storage format under [" + IHNF + "] index type, but got " + faissIndex.getIndexType());
        }
        return faissHNSWFlatIndex;
    }

    @Override
    public String getIndexType() {
        return IHNF;
    }
}

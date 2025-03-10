/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;

import java.io.IOException;

/**
 * This searcher directly reads FAISS index file via the provided {@link IndexInput} then perform vector search on it.
 */
public class FaissMemoryOptimizedSearcher implements VectorSearcher {
    private final IndexInput indexInput;
    private FaissIndex faissIndex;

    public FaissMemoryOptimizedSearcher(IndexInput indexInput) throws IOException {
        this.indexInput = indexInput;
        try {
            this.faissIndex = FaissIndex.load(indexInput);
        } catch (UnsupportedFaissIndexException e) {
            // TODO(KDY) : Remove this in part-7 (Complete Faiss Loading Part at Codec Level)
        }
    }

    @Override
    public void search(float[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {
        // TODO(KDY) : This will be covered in part-7 (Complete Faiss Loading Part at Codec Level)
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public void search(byte[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {
        // TODO(KDY) : This will be covered in part-7 (Complete Faiss Loading Part at Codec Level)
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public void close() throws IOException {
        indexInput.close();
    }
}

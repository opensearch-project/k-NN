/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.ReadAdvice;
import org.apache.lucene.util.IOUtils;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;
import org.opensearch.knn.memoryoptsearch.VectorSearcherFactory;

import java.io.IOException;

/**
 * This factory returns {@link VectorSearcher} that performs vector search directly on FAISS index.
 * Note that we pass `RANDOM` as advice to prevent the underlying storage from performing read-ahead. Since vector search naturally accesses
 * random vector locations, read-ahead does not improve performance. By passing the `RANDOM` context, we explicitly indicate that
 * this searcher will access vectors randomly.
 */
public class FaissMemoryOptimizedSearcherFactory implements VectorSearcherFactory {
    @Override
    public VectorSearcher createVectorSearcher(final Directory directory, final String fileName) throws IOException {
        final IndexInput indexInput = directory.openInput(
            fileName,
            new IOContext(IOContext.Context.DEFAULT, null, null, ReadAdvice.RANDOM)
        );

        try {
            // Try load it. Not all FAISS index types are currently supported at the moment.
            return new FaissMemoryOptimizedSearcher(indexInput);
        } catch (UnsupportedFaissIndexException e) {
            // Clean up input stream.
            try {
                IOUtils.close(indexInput);
            } catch (IOException ioException) {}
            return null;
        }
    }
}

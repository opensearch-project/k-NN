/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.FieldInfo;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
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
@Log4j2
public class FaissMemoryOptimizedSearcherFactory implements VectorSearcherFactory {

    @Override
    public VectorSearcher createVectorSearcher(
        final Directory directory,
        final String fileName,
        final FieldInfo fieldInfo,
        final IOContext ioContext,
        final FlatVectorsReader flatVectorsReader
    ) throws IOException {
        final IndexInput indexInput = directory.openInput(fileName, ioContext);

        try {
            // Try load it. Not all FAISS index types are currently supported at the moment.
            FaissIndex faissIndex = FaissIndex.load(indexInput);
            maybeSetFlatVectorsFromReader(faissIndex, flatVectorsReader, fieldInfo);
            return new FaissMemoryOptimizedSearcher(indexInput, fieldInfo, faissIndex, flatVectorsReader.getFlatVectorScorer());
        } catch (UnsupportedFaissIndexException e) {
            // Clean up input stream.
            try {
                IOUtils.close(indexInput);
            } catch (IOException ioException) {}

            throw e;
        }
    }

    /**
     * When the FAISS HNSW index has no flat vector storage (e.g., BBQ), wire in the provided
     * {@link FlatVectorsReader} as the flat vector source.
     */
    private void maybeSetFlatVectorsFromReader(
        final FaissIndex faissIndex,
        final FlatVectorsReader flatVectorsReader,
        final FieldInfo fieldInfo
    ) {
        if (faissIndex instanceof FaissIdMapIndex idMapIndex) {
            final FaissIndex nested = idMapIndex.getNestedIndex();
            if (nested instanceof AbstractFaissHNSWIndex hnswIndex && hnswIndex.getFlatVectors() == null) {
                hnswIndex.setFlatVectors(new FaissBBQFlatIndex(flatVectorsReader, fieldInfo.getName()));
            }
        }
    }
}

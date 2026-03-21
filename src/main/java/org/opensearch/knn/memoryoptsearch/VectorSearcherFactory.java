/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;

import java.io.IOException;

/**
 * Factory to create {@link VectorSearcher}.
 * Provided parameters will have {@link Directory} and a file name where implementation can rely on it to open an input stream.
 */
public interface VectorSearcherFactory {

    /**
     * Create a non-null {@link VectorSearcher} with given Lucene's {@link Directory}.
     * <p>
     * The {@code flatVectorsReader} provides access to Lucene's flat vector storage. For Faiss SQ (for 1 bit) indices
     * where Faiss skips flat storage via {@code IO_FLAG_SKIP_STORAGE}, the reader is used to wire in
     * a {@code FaissScalarQuantizedFlatIndex} backed by Lucene's quantized reader.
     *
     * @param directory Lucene's Directory.
     * @param fileName Logical file name to load.
     * @param fieldInfo Field info containing metadata for ADC extraction
     * @param ioContext IOContext to use when opening the file
     * @param flatVectorsReader Reader providing flat vector scoring and storage
     * @return Null instance if it is not supported, otherwise return {@link VectorSearcher}
     * @throws IOException
     */
    VectorSearcher createVectorSearcher(
        Directory directory,
        String fileName,
        FieldInfo fieldInfo,
        IOContext ioContext,
        FlatVectorsReader flatVectorsReader
    ) throws IOException;
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.IOUtils;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsReader;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;
import org.opensearch.knn.memoryoptsearch.faiss.FaissMemoryOptimizedSearcher;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Reader for Faiss 104 scalar quantized vector fields. Extends {@link NativeEngines990KnnVectorsReader}
 * and overrides search behavior to always force memory-optimized search regardless of the
 * index-level setting.
 *
 * <p>Key differences from the parent:
 * <ul>
 *   <li>Always forces memory-optimized search — not gated by index setting</li>
 *   <li>No quantization state cache (quantization is handled by Lucene, not k-NN's framework)</li>
 *   <li>No NativeMemoryCacheManager invalidation on close</li>
 *   <li>Byte vector search is not supported</li>
 * </ul>
 *
 * <p>{@link #getFloatVectorValues(String)} delegates to Lucene's
 * {@code Lucene104ScalarQuantizedVectorsReader}, which returns a {@link FloatVectorValues}
 * with both {@code scorer()} (quantized) and {@code rescorer()} (full-precision) support.
 */
@Log4j2
public class Faiss1040ScalarQuantizedKnnVectorsReader extends NativeEngines990KnnVectorsReader {

    Faiss1040ScalarQuantizedKnnVectorsReader(SegmentReadState state, FlatVectorsReader flatVectorsReader) {
        super(state, flatVectorsReader);
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
        throw new UnsupportedOperationException("Byte vector search is not supported for Faiss scalar quantized format");
    }

    /**
     * Always uses memory-optimized search — not gated by the index-level memory_optimized_search
     * setting. A null target triggers warmup initialization.
     * Throws IllegalStateException if the searcher cannot be loaded (e.g., no native file).
     */
    @Override
    public void search(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        // Null target is the warmup signal — handle before attempting searcher load
        if (target == null) {
            throw new FaissMemoryOptimizedSearcher.WarmupInitializationException("Null vector supplied for warmup");
        }

        final FieldInfo fieldInfo = segmentReadState.fieldInfos.fieldInfo(field);
        final VectorSearcher memoryOptimizedSearcher = loadMemoryOptimizedSearcherIfRequired(fieldInfo);

        if (memoryOptimizedSearcher == null) {
            throw new IllegalStateException(
                "Faiss scalar quantized format requires memory optimized search but searcher could not be loaded for field [" + field + "]"
            );
        }

        memoryOptimizedSearcher.search(target, knnCollector, acceptDocs);
    }

    @Override
    public void search(String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        throw new UnsupportedOperationException("Byte vector search is not supported for Faiss scalar quantized format");
    }

    /**
     * No NativeMemoryCacheManager invalidation or quantization state cache cleanup needed —
     * this format doesn't use the k-NN quantization framework or the native memory cache.
     */
    @Override
    public void close() throws IOException {
        final List<Closeable> closeables = new ArrayList<>();
        closeables.add(flatVectorsReader);
        if (vectorSearcherHolder != null) {
            closeables.add(vectorSearcherHolder.getVectorSearcher());
        }
        IOUtils.close(closeables);
    }
}

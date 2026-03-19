/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.KnnCollector;
import org.opensearch.knn.index.codec.nativeindex.AbstractNativeEnginesKnnVectorsReader;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsReader;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;
import org.opensearch.knn.memoryoptsearch.faiss.FaissMemoryOptimizedSearcher;

import java.io.IOException;

/**
 * Reader for Faiss 1040 scalar quantized vector fields. Extends {@link AbstractNativeEnginesKnnVectorsReader}
 * and always forces memory-optimized search regardless of the index-level setting.
 *
 * <p>Key differences from {@link NativeEngines990KnnVectorsReader}:
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
public class Faiss1040ScalarQuantizedKnnVectorsReader extends AbstractNativeEnginesKnnVectorsReader {
    private volatile FieldInfo fieldInfo;

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
        if (this.fieldInfo == null) {
            synchronized (vectorSearcherHolderLockObject) {
                if (this.fieldInfo == null) {
                    this.fieldInfo = segmentReadState.fieldInfos.fieldInfo(field);
                }
            }
        }
        final VectorSearcher memoryOptimizedSearcher = loadMemoryOptimizedSearcherIfRequired(this.fieldInfo);

        // On warmup, target is null. We load the searcher first to trigger memory-mapping of vectors(partial load),
        // then throw instead of returning silently so the warmup call is detectable and can be validated in tests.
        // TODO: Support MOS warmup properly
        if (target == null) {
            throw new FaissMemoryOptimizedSearcher.WarmupInitializationException("Null vector supplied for warmup");
        }

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
}

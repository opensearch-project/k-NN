/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import com.google.common.annotations.VisibleForTesting;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsReader;
import org.opensearch.knn.index.codec.nativeindex.AbstractNativeEnginesKnnVectorsReader;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.util.WarmupUtil;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;

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
    Faiss1040ScalarQuantizedKnnVectorsReader(SegmentReadState state, FlatVectorsReader flatVectorsReader) {
        super(state, flatVectorsReader);
    }

    @VisibleForTesting
    FlatVectorsReader getFlatVectorsReader() {
        return flatVectorsReader;
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
        final VectorSearcher memoryOptimizedSearcher = loadMemoryOptimizedSearcherIfRequired(fieldInfos.fieldInfo(field));

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
     * Warms up the on-disk data for the given scalar-quantized field.
     * <p>
     * The full-precision vectors are always warmed by reading each vector explicitly through the underlying
     * {@link ScalarQuantizedFloatVectorValues} (they cannot be warmed through {@link WarmupUtil}, whose slice
     * is backed by the quantized data).
     * <p>
     * When a native engine (.faiss) graph file is present, the HNSW graph and the quantized vectors are warmed
     * via the memory-optimized searcher. When the graph was skipped by the approximate threshold there is no
     * .faiss file - search is served by exact search over the quantized {@code .veq} codes, so those are warmed
     * directly through the flat reader's index slice instead of loading a (non-existent) searcher.
     *
     * @param fieldName the name of the vector field to warm up
     * @throws IOException if an I/O error occurs while reading the underlying data
     */
    @Override
    public void warmUp(final String fieldName) throws IOException {
        // Warm up full-precision vectors
        // We cannot rely on WarmupUtil, which extracts the IndexInput from vector values and reads through it.
        // Because, the IndexInput returned by vector values is backed by quantized vectors.
        // Therefore, to warm up full-precision vectors, we need to load them explicitly as below.
        final ScalarQuantizedFloatVectorValues vectorValues = (ScalarQuantizedFloatVectorValues) flatVectorsReader.getFloatVectorValues(
            fieldName
        );
        // This would mean that vectors are not present for this segment hence we should not proceed in warmup setup.
        if (vectorValues == null || vectorValues.size() == 0) {
            log.info("No vectors present in the segment {} for field {}", this.segmentReadState.segmentInfo.name, fieldName);
            return;
        }

        for (int i = 0; i < vectorValues.size(); ++i) {
            vectorValues.vectorValue(i);
        }

        // When the graph was skipped by the approximate threshold there is no native engine (.faiss) file, so
        // there is no memory-optimized searcher to load. In that case search is served by exact search over the
        // quantized .veq codes, so warm those directly through the flat reader's index slice. Only attempt the
        // memory-optimized searcher when a graph file is actually present; that path keeps warming the .veq codes
        // and the graph, and still surfaces the error/warn logs for a genuine load failure.
        final FieldInfo fieldInfo = fieldInfos.fieldInfo(fieldName);
        final boolean hasGraphFile = KNNCodecUtil.getNativeEngineFileFromFieldInfo(fieldInfo, segmentReadState.segmentInfo) != null;
        if (hasGraphFile == false) {
            // Warm the quantized .veq codes directly through the flat reader's index slice
            // (ScalarQuantizedFloatVectorValues exposes them via HasIndexSlice#getSlice).
            final IndexInput veqSlice = vectorValues.getSlice();
            if (veqSlice != null) {
                WarmupUtil.readAll(veqSlice);
            }
            return;
        }

        final VectorSearcher memoryOptimizedSearcher = loadMemoryOptimizedSearcherIfRequired(fieldInfo);
        if (memoryOptimizedSearcher != null) {
            // MOS is supported, warm up search parts
            memoryOptimizedSearcher.warmUp();
        } else {
            log.warn("Memory optimized search is not supported for {}", fieldName);
        }
    }
}

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
     * Warms three files: the quantized codes in {@code .veq}, the full-precision floats in
     * {@code .vec}, and (when present) the FAISS HNSW graph in {@code .faiss}.
     * <p>
     * The {@code .veq} slice is warmed here via the quantized delegate — it's the one file
     * {@link VectorSearcher#warmUp()} does not cover on its own. When a {@code .faiss} graph
     * file is present, {@link VectorSearcher#warmUp()} bulk-reads the graph and then iterates
     * the fp32 float values, which touches every {@code .vec} page via this wrapper's
     * {@link ScalarQuantizedFloatVectorValues#getFloatVectorValues() fp32 delegate} — so we
     * don't duplicate that work.
     * <p>
     * When the graph was skipped by the approximate threshold there is no {@code .faiss} file
     * and no memory-optimized searcher to load. Exact search over the just-warmed {@code .veq}
     * codes serves queries, and we additionally warm the {@code .vec} fp32 floats directly
     * through the fp32 delegate for rescoring.
     *
     * @param fieldName the name of the vector field to warm up
     * @throws IOException if an I/O error occurs while reading the underlying data
     */
    @Override
    public void warmUp(final String fieldName) throws IOException {
        final ScalarQuantizedFloatVectorValues vectorValues = (ScalarQuantizedFloatVectorValues) flatVectorsReader.getFloatVectorValues(
            fieldName
        );
        // This would mean that vectors are not present for this segment hence we should not proceed in warmup setup.
        if (vectorValues == null || vectorValues.size() == 0) {
            log.info("No vectors present in the segment {} for field {}", this.segmentReadState.segmentInfo.name, fieldName);
            return;
        }

        // Warm up the .veq (quantized codes). This is the only file MOS's
        // warmUp() does not cover — it handles .faiss (graph) and .vec (fp32).
        if (vectorValues.getQuantizedVectorValues() != null) {
            WarmupUtil.readAll(vectorValues.getQuantizedVectorValues());
        }

        // When the approximate threshold skipped the HNSW build there's no .faiss file and no
        // memory-optimized searcher to load. Warm the .vec fp32 floats directly through the fp32
        // delegate (its OffHeapFloatVectorValues doesn't implement HasIndexSlice, so readAll falls
        // through to the per-ord loop) and return — exact search over the just-warmed .veq codes
        // serves queries.
        final FieldInfo fieldInfo = fieldInfos.fieldInfo(fieldName);
        final boolean hasGraphFile = KNNCodecUtil.getNativeEngineFileFromFieldInfo(fieldInfo, segmentReadState.segmentInfo) != null;
        if (!hasGraphFile) {
            WarmupUtil.readAll(vectorValues.getFloatVectorValues());
            return;
        }

        final VectorSearcher memoryOptimizedSearcher = loadMemoryOptimizedSearcherIfRequired(fieldInfo);
        if (memoryOptimizedSearcher != null) {
            // Warms the .faiss graph and the .vec fp32 floats.
            memoryOptimizedSearcher.warmUp();
        } else {
            log.warn("Memory optimized search is not supported for {}", fieldName);
        }
    }
}

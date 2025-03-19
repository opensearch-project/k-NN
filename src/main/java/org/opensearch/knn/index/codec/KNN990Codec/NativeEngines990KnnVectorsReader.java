/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOSupplier;
import org.apache.lucene.util.IOUtils;
import org.opensearch.common.UUIDs;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.codec.util.NativeMemoryCacheKeyHelper;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;
import org.opensearch.knn.memoryoptsearch.VectorSearcherFactory;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationStateCacheManager;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationStateReadConfig;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapper.KNN_FIELD;

/**
 * Vectors reader class for reading the flat vectors for native engines. The class provides methods for iterating
 * over the vectors and retrieving their values.
 */
@Log4j2
public class NativeEngines990KnnVectorsReader extends KnnVectorsReader {
    private static final int RESERVE_TWICE_SPACE = 2;
    private static final float SUFFICIENT_LOAD_FACTOR = 0.6f;

    private final FlatVectorsReader flatVectorsReader;
    private Map<String, String> quantizationStateCacheKeyPerField;
    private SegmentReadState segmentReadState;
    private final List<String> cacheKeys;
    private volatile Map<String, VectorSearcher> vectorSearchers;

    public NativeEngines990KnnVectorsReader(final SegmentReadState state, final FlatVectorsReader flatVectorsReader) {
        this.flatVectorsReader = flatVectorsReader;
        this.segmentReadState = state;
        this.cacheKeys = getVectorCacheKeysFromSegmentReaderState(state);
        this.vectorSearchers = null;

        loadCacheKeyMap();

        // Memory optimized searcher will be ONLY used for searching, we don't need this during merging.
        // TODO(KDY) : Enable this based on setting flag value. e.g. index.knn.memory_optimized_search = True
        if (state.context.context() != IOContext.Context.MERGE) {
            loadMemoryOptimizedSearcher();
        }
    }

    private IOSupplier<VectorSearcher> getIndexFileNameIfMemoryOptimizedSearchSupported(final FieldInfo fieldInfo) {
        // Skip non-knn fields.
        final Map<String, String> attributes = fieldInfo.attributes();
        if (attributes == null || attributes.containsKey(KNN_FIELD) == false) {
            return null;
        }

        // Get engine
        final String engineName = attributes.getOrDefault(KNN_ENGINE, KNNEngine.DEFAULT.getName());
        final KNNEngine knnEngine = KNNEngine.getEngine(engineName);

        // Get memory optimized searcher from engine
        final VectorSearcherFactory searcherFactory = knnEngine.getVectorSearcherFactory();
        if (searcherFactory == null) {
            // It's not supported
            return null;
        }

        // Start creating searcher
        final String fileName = KNNCodecUtil.getNativeEngineFileFromFieldInfo(fieldInfo, segmentReadState.segmentInfo);
        if (fileName != null) {
            return () -> searcherFactory.createVectorSearcher(segmentReadState.directory, fileName);
        }

        // Not supported
        return null;
    }

    private synchronized void loadMemoryOptimizedSearcher() {
        if (vectorSearchers != null) {
            return;
        }

        final Map<String, VectorSearcher> vectorSearcherPerField = new HashMap<>(
            RESERVE_TWICE_SPACE * segmentReadState.fieldInfos.size(),
            SUFFICIENT_LOAD_FACTOR
        );

        try {
            for (FieldInfo fieldInfo : segmentReadState.fieldInfos) {
                final IOSupplier<VectorSearcher> searcherSupplier = getIndexFileNameIfMemoryOptimizedSearchSupported(fieldInfo);
                if (searcherSupplier != null) {
                    final VectorSearcher searcher = searcherSupplier.get();
                    if (searcher != null) {
                        // It's supported. There can be a case where a certain index type underlying is not yet supported while KNNEngine
                        // itself supports memory optimized searching.
                        vectorSearcherPerField.put(fieldInfo.getName(), searcher);
                    }
                }
            }

            vectorSearchers = vectorSearcherPerField;
        } catch (Exception e) {
            // Close opened searchers first, then suppress
            try {
                IOUtils.closeWhileHandlingException(vectorSearcherPerField.values());
            } catch (Exception closeException) {
                log.error(closeException.getMessage(), closeException);
            }
            throw new RuntimeException(e);
        }
    }

    /**
     * Checks consistency of this reader.
     *
     * <p>Note that this may be costly in terms of I/O, e.g. may involve computing a checksum value
     * against large data files.
     *
     */
    @Override
    public void checkIntegrity() throws IOException {
        flatVectorsReader.checkIntegrity();
    }

    /**
     * Returns the {@link FloatVectorValues} for the given {@code field}. The behavior is undefined if
     * the given field doesn't have KNN vectors enabled on its {@link FieldInfo}. The return value is
     * never {@code null}.
     *
     * @param field {@link String}
     */
    @Override
    public FloatVectorValues getFloatVectorValues(final String field) throws IOException {
        return flatVectorsReader.getFloatVectorValues(field);
    }

    /**
     * Returns the {@link ByteVectorValues} for the given {@code field}. The behavior is undefined if
     * the given field doesn't have KNN vectors enabled on its {@link FieldInfo}. The return value is
     * never {@code null}.
     *
     * @param field {@link String}
     */
    @Override
    public ByteVectorValues getByteVectorValues(final String field) throws IOException {
        return flatVectorsReader.getByteVectorValues(field);
    }

    /**
     * Return the k nearest neighbor documents as determined by comparison of their vector values for
     * this field, to the given vector, by the field's similarity function. The score of each document
     * is derived from the vector similarity in a way that ensures scores are positive and that a
     * larger score corresponds to a higher ranking.
     *
     * <p>The search is allowed to be approximate, meaning the results are not guaranteed to be the
     * true k closest neighbors. For large values of k (for example when k is close to the total
     * number of documents), the search may also retrieve fewer than k documents.
     *
     * <p>The returned {@link TopDocs} will contain a {@link ScoreDoc} for each nearest neighbor, in
     * order of their similarity to the query vector (decreasing scores). The {@link TotalHits}
     * contains the number of documents visited during the search. If the search stopped early because
     * it hit {@code visitedLimit}, it is indicated through the relation {@code
     * TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO}.
     *
     * <p>The behavior is undefined if the given field doesn't have KNN vectors enabled on its {@link
     * FieldInfo}. The return value is never {@code null}.
     *
     * @param field        the vector field to search
     * @param target       the vector-valued query
     * @param knnCollector a KnnResults collector and relevant settings for gathering vector results
     * @param acceptDocs   {@link Bits} that represents the allowed documents to match, or {@code null}
     *                     if they are all allowed to match.
     */
    @Override
    public void search(String field, float[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {
        // TODO: This is a temporary hack where we are using KNNCollector to initialize the quantization state.
        if (knnCollector instanceof QuantizationConfigKNNCollector) {
            String cacheKey = quantizationStateCacheKeyPerField.get(field);
            FieldInfo fieldInfo = segmentReadState.fieldInfos.fieldInfo(field);
            QuantizationState quantizationState = QuantizationStateCacheManager.getInstance()
                .getQuantizationState(
                    new QuantizationStateReadConfig(
                        segmentReadState,
                        QuantizationService.getInstance().getQuantizationParams(fieldInfo),
                        field,
                        cacheKey
                    )
                );
            ((QuantizationConfigKNNCollector) knnCollector).setQuantizationState(quantizationState);
            return;
        }

        if (trySearchWithMemoryOptimizedSearch(field, target, knnCollector, acceptDocs, true)) {
            return;
        }

        throw new UnsupportedOperationException("Search functionality using codec is not supported with Native Engine Reader");
    }

    /**
     * Return the k nearest neighbor documents as determined by comparison of their vector values for
     * this field, to the given vector, by the field's similarity function. The score of each document
     * is derived from the vector similarity in a way that ensures scores are positive and that a
     * larger score corresponds to a higher ranking.
     *
     * <p>The search is allowed to be approximate, meaning the results are not guaranteed to be the
     * true k closest neighbors. For large values of k (for example when k is close to the total
     * number of documents), the search may also retrieve fewer than k documents.
     *
     * <p>The returned {@link TopDocs} will contain a {@link ScoreDoc} for each nearest neighbor, in
     * order of their similarity to the query vector (decreasing scores). The {@link TotalHits}
     * contains the number of documents visited during the search. If the search stopped early because
     * it hit {@code visitedLimit}, it is indicated through the relation {@code
     * TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO}.
     *
     * <p>The behavior is undefined if the given field doesn't have KNN vectors enabled on its {@link
     * FieldInfo}. The return value is never {@code null}.
     *
     * @param field        the vector field to search
     * @param target       the vector-valued query
     * @param knnCollector a KnnResults collector and relevant settings for gathering vector results
     * @param acceptDocs   {@link Bits} that represents the allowed documents to match, or {@code null}
     *                     if they are all allowed to match.
     */
    @Override
    public void search(String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {
        if (trySearchWithMemoryOptimizedSearch(field, target, knnCollector, acceptDocs, false)) {
            return;
        }

        throw new UnsupportedOperationException("Search functionality using codec is not supported with Native Engine Reader");
    }

    /**
     * Closes this stream and releases any system resources associated
     * with it. If the stream is already closed then invoking this
     * method has no effect.
     *
     * <p> As noted in {@link AutoCloseable#close()}, cases where the
     * close may fail require careful attention. It is strongly advised
     * to relinquish the underlying resources and to internally
     * <em>mark</em> the {@code Closeable} as closed, prior to throwing
     * the {@code IOException}.
     *
     * @throws IOException if an I/O error occurs
     */
    @Override
    public void close() throws IOException {
        // Clean up allocated vector indices resources from cache.
        final NativeMemoryCacheManager nativeMemoryCacheManager = NativeMemoryCacheManager.getInstance();
        cacheKeys.forEach(nativeMemoryCacheManager::invalidate);

        // Close a reader.
        List<Closeable> closeables = new ArrayList<>();
        closeables.add(flatVectorsReader);

        // Close vector searchers if loaded.
        if (vectorSearchers != null) {
            closeables.addAll(vectorSearchers.values());
        }

        IOUtils.close(closeables);

        // Clean up quantized state cache.
        if (quantizationStateCacheKeyPerField != null) {
            final QuantizationStateCacheManager quantizationStateCacheManager = QuantizationStateCacheManager.getInstance();
            for (String cacheKey : quantizationStateCacheKeyPerField.values()) {
                quantizationStateCacheManager.evict(cacheKey);
            }
        }
    }

    private boolean trySearchWithMemoryOptimizedSearch(
        String field,
        Object target,
        KnnCollector knnCollector,
        Bits acceptDocs,
        boolean isFloatVector
    ) throws IOException {
        if (vectorSearchers == null) {
            // Null `vectorSearchers` indicates that this reader was instantiated during merge.
            // In this case, we load the searcher on demand for searching.
            // This will not likely happen, by the time searching, this old segment will be merged away anyway.
            loadMemoryOptimizedSearcher();
        }

        // Try with memory optimized searcher
        final VectorSearcher memoryOptimizedSearcher = vectorSearchers.get(field);
        if (memoryOptimizedSearcher != null) {
            if (isFloatVector) {
                memoryOptimizedSearcher.search((float[]) target, knnCollector, acceptDocs);
            } else {
                memoryOptimizedSearcher.search((byte[]) target, knnCollector, acceptDocs);
            }
            return true;
        }
        return false;
    }

    private void loadCacheKeyMap() {
        quantizationStateCacheKeyPerField = new HashMap<>();
        for (FieldInfo fieldInfo : segmentReadState.fieldInfos) {
            String cacheKey = UUIDs.base64UUID();
            quantizationStateCacheKeyPerField.put(fieldInfo.getName(), cacheKey);
        }
    }

    private static List<String> getVectorCacheKeysFromSegmentReaderState(SegmentReadState segmentReadState) {
        final List<String> cacheKeys = new ArrayList<>();

        for (FieldInfo field : segmentReadState.fieldInfos) {
            final String vectorIndexFileName = KNNCodecUtil.getNativeEngineFileFromFieldInfo(field, segmentReadState.segmentInfo);
            if (vectorIndexFileName == null) {
                continue;
            }
            final String cacheKey = NativeMemoryCacheKeyHelper.constructCacheKey(vectorIndexFileName, segmentReadState.segmentInfo);
            cacheKeys.add(cacheKey);
        }

        return cacheKeys;
    }
}

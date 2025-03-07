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

import lombok.extern.slf4j.Slf4j;
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

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapper.KNN_FIELD;

/**
 * Vectors reader class for reading the flat vectors for native engines. The class provides methods for iterating
 * over the vectors and retrieving their values.
 */
@Slf4j
public class NativeEngines990KnnVectorsReader extends KnnVectorsReader {
    private static final int RESERVE_TWICE_SPACE = 2;
    private static final float SUFFICIENT_LOAD_FACTOR = 0.6f;

    private final FlatVectorsReader flatVectorsReader;
    private Map<String, String> quantizationStateCacheKeyPerField;
    private SegmentReadState segmentReadState;
    private final List<String> cacheKeys;
    private Map<String, VectorSearcher> vectorSearchers;

    public NativeEngines990KnnVectorsReader(final SegmentReadState state, final FlatVectorsReader flatVectorsReader) {
        this.flatVectorsReader = flatVectorsReader;
        this.segmentReadState = state;
        this.cacheKeys = getVectorCacheKeysFromSegmentReaderState(state);
        this.vectorSearchers = new HashMap<>(RESERVE_TWICE_SPACE * segmentReadState.fieldInfos.size(), SUFFICIENT_LOAD_FACTOR);
        loadCacheKeyMap();

        //
        // TMP(KDY) : Dynamic update will be covered in part-7. Please refer to
        // https://github.com/opensearch-project/k-NN/issues/2401#issuecomment-2699777824
        //
        final boolean isMemoryOptimizedSearchEnabled = false;
        if (isMemoryOptimizedSearchEnabled) {
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

    private void loadMemoryOptimizedSearcher() {
        try {
            for (FieldInfo fieldInfo : segmentReadState.fieldInfos) {
                final IOSupplier<VectorSearcher> searcherSupplier = getIndexFileNameIfMemoryOptimizedSearchSupported(fieldInfo);
                if (searcherSupplier != null) {
                    final VectorSearcher searcher = Objects.requireNonNull(searcherSupplier.get());
                    vectorSearchers.put(fieldInfo.getName(), searcher);
                }
            }
        } catch (Exception e) {
            // Close opened searchers first, then suppress
            try {
                IOUtils.closeWhileHandlingException(vectorSearchers.values());
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

        // Try with memory optimized searcher
        final VectorSearcher memoryOptimizedSearcher = vectorSearchers.get(field);
        if (memoryOptimizedSearcher != null) {
            memoryOptimizedSearcher.search(target, knnCollector, acceptDocs);
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
        IOUtils.close(flatVectorsReader);

        // Clean up quantized state cache.
        if (quantizationStateCacheKeyPerField != null) {
            final QuantizationStateCacheManager quantizationStateCacheManager = QuantizationStateCacheManager.getInstance();
            for (String cacheKey : quantizationStateCacheKeyPerField.values()) {
                quantizationStateCacheManager.evict(cacheKey);
            }
        }

        // TODO(KDY)
        // Close all memory optimized searchers.
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

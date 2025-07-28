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

import lombok.Getter;
import lombok.NonNull;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOSupplier;
import org.apache.lucene.util.IOUtils;
import org.opensearch.common.UUIDs;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.codec.util.NativeMemoryCacheKeyHelper;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.memory.NativeMemoryEntryContext;
import org.opensearch.knn.index.memory.NativeMemoryLoadStrategy;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;
import org.opensearch.knn.memoryoptsearch.VectorSearcherFactory;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationStateCacheManager;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationStateReadConfig;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapper.KNN_FIELD;
import static org.opensearch.knn.index.util.IndexUtil.getParametersAtLoading;

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
    private final SegmentReadState segmentReadState;
    private final List<String> cacheKeys;
    private volatile Map<String, VectorSearcherHolder> vectorSearchers;

    public NativeEngines990KnnVectorsReader(final SegmentReadState state, final FlatVectorsReader flatVectorsReader) {
        this.flatVectorsReader = flatVectorsReader;
        this.segmentReadState = state;
        this.cacheKeys = getVectorCacheKeysFromSegmentReaderState(state);

        loadCacheKeyMap();
        fillVectorSearcherTable();
        warmUpSegmentIfRequired();
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
                        QuantizationService.getInstance().getQuantizationParams(fieldInfo, segmentReadState.segmentInfo.getVersion()),
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
        final List<Closeable> closeables = new ArrayList<>();
        closeables.add(flatVectorsReader);

        // Close vector searchers if loaded.
        if (vectorSearchers != null) {
            closeables.addAll(
                vectorSearchers.values().stream().filter(VectorSearcherHolder::isSet).map(VectorSearcherHolder::getVectorSearcher).toList()
            );
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
        final String field,
        final Object target,
        final KnnCollector knnCollector,
        final Bits acceptDocs,
        final boolean isFloatVector
    ) throws IOException {
        // Try with memory optimized searcher
        final VectorSearcher memoryOptimizedSearcher = loadMemoryOptimizedSearcherIfRequired(field);

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

    private void fillVectorSearcherTable() {
        // We need sufficient memory space for this table as it will be queried for every single search.
        // Hence, having larger space to approximate a perfect hash here.
        vectorSearchers = new HashMap<>(RESERVE_TWICE_SPACE * segmentReadState.fieldInfos.size(), SUFFICIENT_LOAD_FACTOR);

        for (final FieldInfo fieldInfo : segmentReadState.fieldInfos) {
            final IOSupplier<VectorSearcher> searcherIOSupplier = getVectorSearcherSupplier(fieldInfo);
            if (searcherIOSupplier != null) {
                // This field type is supported
                vectorSearchers.put(fieldInfo.getName(), new VectorSearcherHolder());
            }
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

    private VectorSearcher loadMemoryOptimizedSearcherIfRequired(final String fieldName) {
        final VectorSearcherHolder searcherHolder = vectorSearchers.get(fieldName);
        if (searcherHolder == null) {
            // This is not KNN field or unsupported field.
            return null;
        }

        if (searcherHolder.isSet()) {
            return searcherHolder.getVectorSearcher();
        }

        synchronized (searcherHolder) {
            if (searcherHolder.isSet()) {
                return searcherHolder.getVectorSearcher();
            }

            VectorSearcher searcher = null;

            try {
                final FieldInfo fieldInfo = segmentReadState.fieldInfos.fieldInfo(fieldName);
                if (fieldInfo != null) {
                    final IOSupplier<VectorSearcher> searcherSupplier = getVectorSearcherSupplier(fieldInfo);
                    if (searcherSupplier != null) {
                        searcher = searcherSupplier.get();
                        if (searcher != null) {
                            // It's supported. There can be a case where a certain index type underlying is not yet supported while
                            // KNNEngine
                            // itself supports memory optimized searching.
                            searcherHolder.setVectorSearcher(searcher);
                        }
                    }
                }

                return searcher;
            } catch (Exception e) {
                // Close opened searchers first, then suppress
                try {
                    IOUtils.closeWhileHandlingException(searcher);
                } catch (Exception closeException) {
                    log.error(closeException.getMessage(), closeException);
                }
                throw new RuntimeException(e);
            }
        }
    }

    private IOSupplier<VectorSearcher> getVectorSearcherSupplier(final FieldInfo fieldInfo) {
        // Skip non-knn fields.
        final Map<String, String> attributes = fieldInfo.attributes();
        if (attributes == null || attributes.containsKey(KNN_FIELD) == false) {
            return null;
        }

        // Try to get KNN engine from fieldInfo.
        final KNNEngine knnEngine = FieldInfoExtractor.extractKNNEngine(fieldInfo);

        // No KNNEngine is available
        if (knnEngine == null) {
            return null;
        }

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

    private void warmUpSegmentIfRequired() {
        SegmentInfo segmentInfo = segmentReadState.segmentInfo;
        if (NativeEngineSegmentAttributeParser.parseWarmup(segmentInfo)) {
            Set<String> memoryOptimizedFieldNames = NativeEngineSegmentAttributeParser.parseMemoryOptimizedFields(segmentInfo);
            String indexName = NativeEngineSegmentAttributeParser.parseIndexName(segmentInfo);
            for (final FieldInfo fieldInfo : segmentReadState.fieldInfos) {
                if (memoryOptimizedFieldNames.contains(fieldInfo.getName())) {
                    String dataTypeStr = fieldInfo.getAttribute(VECTOR_DATA_TYPE_FIELD);
                    if (dataTypeStr == null) {
                        continue;
                    }
                    try {
                        boolean isFloat = VectorDataType.get(dataTypeStr) == VectorDataType.FLOAT;
                        trySearchWithMemoryOptimizedSearch(
                            fieldInfo.getName(),
                            null,
                            null,
                            null,
                            isFloat
                        );
                    } catch (Exception e) {
                        log.warn("Failed to warm up memory optimized field: {}", fieldInfo.getName());
                    }
                } else {
                    final String vectorIndexFileName = KNNCodecUtil.getNativeEngineFileFromFieldInfo(fieldInfo, segmentInfo);
                    if (vectorIndexFileName == null) {
                        continue;
                    }
                    final String cacheKey = NativeMemoryCacheKeyHelper.constructCacheKey(vectorIndexFileName, segmentInfo);
                    final NativeMemoryCacheManager cacheManager = NativeMemoryCacheManager.getInstance();
                    try {
                        final String spaceTypeName = fieldInfo.attributes().getOrDefault(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue());
                        final SpaceType spaceType = SpaceType.getSpace(spaceTypeName);
                        final KNNEngine knnEngine = FieldInfoExtractor.extractKNNEngine(fieldInfo);
                        final VectorDataType vectorDataType = FieldInfoExtractor.extractVectorDataType(fieldInfo);
                        final QuantizationParams quantizationParams = QuantizationService.getInstance()
                            .getQuantizationParams(fieldInfo, segmentInfo.getVersion());
                        cacheManager.get(
                            new NativeMemoryEntryContext.IndexEntryContext(
                                segmentInfo.dir,
                                cacheKey,
                                NativeMemoryLoadStrategy.IndexLoadStrategy.getInstance(),
                                getParametersAtLoading(spaceType, knnEngine, indexName, vectorDataType, quantizationParams),
                                indexName
                            ),
                            true
                        );
                    } catch (Exception e) {
                        log.warn("Failed to warm up field: {}", fieldInfo.getName());
                    }
                }
            }
        }
    }

    /**
     * A holder for a {@link VectorSearcher} reference.
     * Initially, the reference is {@code null}. The reference is expected to be set exactly once via the {@code setVectorSearcher} method,
     * following a proper thread-safety policy (In most cases, `synchronized` will work). Once the reference is set,
     * it is assumed to remain immutable.
     */
    public static class VectorSearcherHolder {
        @Getter
        private volatile VectorSearcher vectorSearcher = null;

        /**
         * Updates the {@link VectorSearcher} reference.
         * This method should be called with an appropriate thread-safety mechanism.
         * In most cases, using {@code synchronized} is sufficient.
         *
         * @param vectorSearcher the {@link VectorSearcher} instance to assign.
         */
        public void setVectorSearcher(@NonNull final VectorSearcher vectorSearcher) {
            assert (this.vectorSearcher == null);
            this.vectorSearcher = vectorSearcher;
        }

        public boolean isSet() {
            return vectorSearcher != null;
        }
    }
}

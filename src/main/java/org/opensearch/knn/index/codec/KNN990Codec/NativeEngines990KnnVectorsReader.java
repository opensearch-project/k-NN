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
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.util.Bits;
import org.opensearch.common.UUIDs;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.codec.nativeindex.AbstractNativeEnginesKnnVectorsReader;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.codec.util.NativeMemoryCacheKeyHelper;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.index.util.WarmupUtil;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;
import org.opensearch.knn.memoryoptsearch.faiss.FaissScorableByteVectorValues;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationStateCacheManager;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationStateReadConfig;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.FieldInfoExtractor.hasQuantizationConfig;

/**
 * Vectors reader class for reading the flat vectors for native engines. The class provides methods for iterating
 * over the vectors and retrieving their values.
 */
@Log4j2
public class NativeEngines990KnnVectorsReader extends AbstractNativeEnginesKnnVectorsReader {

    private Map<String, String> quantizationStateCacheKeyPerField;
    private final List<String> cacheKeys;

    public NativeEngines990KnnVectorsReader(final SegmentReadState state, final FlatVectorsReader flatVectorsReader) {
        super(state, flatVectorsReader);
        this.cacheKeys = getVectorCacheKeysFromSegmentReaderState(state);
        loadCacheKeyMap();
    }

    /**
     * Returns a composite {@link FloatVectorValues} that bundles full-precision float vectors with
     * quantized byte vectors when quantization is available. The composite delegates:
     * <ul>
     *   <li>{@code vectorValue()} to full-precision floats (for merge/flush)</li>
     *   <li>{@code scorer()} quantizes the float query and delegates to quantized byte values</li>
     *   <li>{@code rescorer()} to full-precision floats (for full-fidelity rescoring)</li>
     * </ul>
     * Falls back to plain float vector values when quantization is not configured.
     */
    @Override
    public FloatVectorValues getFloatVectorValues(final String field) throws IOException {
        final FloatVectorValues rawFloatVectorValues = flatVectorsReader.getFloatVectorValues(field);
        final FieldInfo fieldInfo = fieldInfos.fieldInfo(field);
        if (fieldInfo.getVectorEncoding() == VectorEncoding.FLOAT32 && hasQuantizationConfig(fieldInfo)) {
            final VectorSearcher vectorSearcher = loadMemoryOptimizedSearcherIfRequired(fieldInfo);
            if (vectorSearcher != null) {
                final ByteVectorValues byteVectorValues = vectorSearcher.getByteVectorValues(rawFloatVectorValues.iterator());
                return new QuantizedFloatVectorValues(rawFloatVectorValues, byteVectorValues, field);
            }
        }
        return rawFloatVectorValues;
    }

    /**
     * Returns the {@link ByteVectorValues} for the given field by delegating to the flat vectors reader.
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
    public void search(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        // TODO: This is a temporary hack where we are using KNNCollector to initialize the quantization state.
        if (knnCollector instanceof QuantizationConfigKNNCollector) {
            ((QuantizationConfigKNNCollector) knnCollector).setQuantizationState(getQuantizationState(field));
            return;
        }

        final FieldInfo fieldInfo = fieldInfos.fieldInfo(field);
        if (trySearchWithMemoryOptimizedSearch(fieldInfo, target, knnCollector, acceptDocs, true)) {
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
    public void search(String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        final FieldInfo fieldInfo = fieldInfos.fieldInfo(field);
        // searching with byte vector is not supported by ADC.
        if (trySearchWithMemoryOptimizedSearch(fieldInfo, target, knnCollector, acceptDocs, false)) {
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

        // To close Reader and Vector Searcher
        super.close();

        // Clean up quantized state cache.
        if (quantizationStateCacheKeyPerField != null) {
            final QuantizationStateCacheManager quantizationStateCacheManager = QuantizationStateCacheManager.getInstance();
            for (String cacheKey : quantizationStateCacheKeyPerField.values()) {
                quantizationStateCacheManager.evict(cacheKey);
            }
        }
    }

    private boolean trySearchWithMemoryOptimizedSearch(
        final FieldInfo fieldInfo,
        final Object target,
        final KnnCollector knnCollector,
        final AcceptDocs acceptDocs,
        final boolean isFloatVector
    ) throws IOException {
        // Try with memory optimized searcher
        final VectorSearcher memoryOptimizedSearcher = loadMemoryOptimizedSearcherIfRequired(fieldInfo);

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
        for (FieldInfo fieldInfo : fieldInfos) {
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

    /**
     * Warms up the on-disk data for the given field by loading the HNSW graph and flat vectors
     * into the OS page cache.
     * <p>
     * For quantized fields (those with a {@code QFRAMEWORK_CONFIG} attribute), this also warms
     * up the full-precision {@code .vec} file via the flat vectors reader.
     *
     * @param fieldName the name of the vector field to warm up
     * @throws IOException if an I/O error occurs while reading the underlying data
     */
    @Override
    public void warmUp(final String fieldName) throws IOException {
        final FieldInfo fieldInfo = fieldInfos.fieldInfo(fieldName);

        final VectorSearcher memoryOptimizedSearcher = loadMemoryOptimizedSearcherIfRequired(fieldInfo);
        if (memoryOptimizedSearcher != null) {
            // For quantized vectors, we should warm up .vec as well.
            if (hasQuantizationConfig(fieldInfo)) {
                WarmupUtil.readAll(flatVectorsReader.getFloatVectorValues(fieldName));
            }

            // Warm up search parts
            memoryOptimizedSearcher.warmUp();
        }
    }

    private QuantizationState getQuantizationState(final String field) throws IOException {
        final String cacheKey = quantizationStateCacheKeyPerField.get(field);
        return QuantizationStateCacheManager.getInstance()
            .getQuantizationState(
                new QuantizationStateReadConfig(
                    segmentReadState,
                    QuantizationService.getInstance().getQuantizationParams(fieldInfos.fieldInfo(field)),
                    field,
                    cacheKey
                )
            );
    }

    /**
     * A composite {@link FloatVectorValues} that bundles full-precision float vectors with
     * quantized byte vectors, following the same pattern as Lucene's {@code ScalarQuantizedVectorValues}.
     * <ul>
     *   <li>{@link #vectorValue(int)} returns full-precision floats (for merge/flush)</li>
     *   <li>{@link #scorer(float[])} quantizes the query and delegates to quantized byte values</li>
     *   <li>{@link #rescorer(float[])} delegates to raw float values (for full-fidelity rescoring)</li>
     * </ul>
     */
    private final class QuantizedFloatVectorValues extends FloatVectorValues {
        private final FloatVectorValues rawFloatVectorValues;
        private final ByteVectorValues quantizedByteVectorValues;
        private final String fieldName;

        QuantizedFloatVectorValues(FloatVectorValues rawFloatVectorValues, ByteVectorValues quantizedByteVectorValues, String fieldName) {
            this.rawFloatVectorValues = rawFloatVectorValues;
            this.quantizedByteVectorValues = quantizedByteVectorValues;
            this.fieldName = fieldName;
        }

        @Override
        public int dimension() {
            return rawFloatVectorValues.dimension();
        }

        @Override
        public int size() {
            return rawFloatVectorValues.size();
        }

        @Override
        public float[] vectorValue(int ord) throws IOException {
            return rawFloatVectorValues.vectorValue(ord);
        }

        @Override
        public QuantizedFloatVectorValues copy() throws IOException {
            return new QuantizedFloatVectorValues(rawFloatVectorValues.copy(), quantizedByteVectorValues.copy(), fieldName);
        }

        @Override
        public DocIndexIterator iterator() {
            return rawFloatVectorValues.iterator();
        }

        @Override
        public int ordToDoc(int ord) {
            return rawFloatVectorValues.ordToDoc(ord);
        }

        @Override
        public Bits getAcceptOrds(Bits acceptDocs) {
            return rawFloatVectorValues.getAcceptOrds(acceptDocs);
        }

        @SuppressWarnings("unchecked")
        @Override
        public VectorScorer scorer(float[] target) throws IOException {
            if (quantizedByteVectorValues instanceof FaissScorableByteVectorValues scorableByteVectorValues
                && FieldInfoExtractor.isAdc(fieldInfos.fieldInfo(fieldName))) {
                // ADC: the FlatVectorsScorer handles float-vs-byte scoring asymmetrically
                return scorableByteVectorValues.scorer(target);
            }
            // Non-ADC: quantize the float query to bytes, then score byte-vs-byte
            final QuantizationState quantizationState = getQuantizationState(fieldName);
            final QuantizationService quantizationService = QuantizationService.getInstance();
            final byte[] quantizedQuery = (byte[]) quantizationService.quantize(
                quantizationState,
                target,
                quantizationService.createQuantizationOutput(quantizationState.getQuantizationParams())
            );
            return quantizedByteVectorValues.scorer(quantizedQuery);
        }

        @Override
        public VectorScorer rescorer(float[] target) throws IOException {
            return rawFloatVectorValues.rescorer(target);
        }
    }
}

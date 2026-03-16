/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.FileDataHint;
import org.apache.lucene.store.FileTypeHint;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.IOSupplier;
import org.apache.lucene.util.IOUtils;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsReader;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;
import org.opensearch.knn.memoryoptsearch.VectorSearcherFactory;
import org.opensearch.knn.memoryoptsearch.faiss.FaissMemoryOptimizedSearcher;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.index.mapper.KNNVectorFieldMapper.KNN_FIELD;

/**
 * Reader for Faiss BBQ vector fields.
 *
 * <p>Key differences from {@link NativeEngines990KnnVectorsReader}:
 * <ul>
 *   <li>Always forces memory-optimized search regardless of the index-level setting</li>
 *   <li>No quantization state cache (BBQ quantization is handled by Lucene, not k-NN's framework)</li>
 *   <li>No NativeMemoryCacheManager invalidation on close</li>
 *   <li>Byte vector search is not supported</li>
 * </ul>
 *
 * <p>{@link #getFloatVectorValues(String)} delegates to Lucene's
 * {@code Lucene104ScalarQuantizedVectorsReader}, which returns a {@link FloatVectorValues}
 * with both {@code scorer()} (quantized BBQ) and {@code rescorer()} (full-precision) support.
 */
@Log4j2
public class FaissBBQ1040KnnVectorsReader extends KnnVectorsReader {

    private final FlatVectorsReader flatVectorsReader;
    private final SegmentReadState segmentReadState;
    // Lazy-initialized; guarded by vectorSearcherHolderLockObject
    private volatile NativeEngines990KnnVectorsReader.VectorSearcherHolder vectorSearcherHolder;
    private final Object vectorSearcherHolderLockObject;
    private final IOContext ioContext;

    FaissBBQ1040KnnVectorsReader(SegmentReadState state, FlatVectorsReader flatVectorsReader) {
        this.flatVectorsReader = flatVectorsReader;
        this.segmentReadState = state;
        this.ioContext = state.context.withHints(FileTypeHint.DATA, FileDataHint.KNN_VECTORS, DataAccessHint.RANDOM);
        this.vectorSearcherHolder = new NativeEngines990KnnVectorsReader.VectorSearcherHolder();
        this.vectorSearcherHolderLockObject = new Object();
    }

    @Override
    public void checkIntegrity() throws IOException {
        flatVectorsReader.checkIntegrity();
    }

    /**
     * Returns float vector values from the underlying Lucene104ScalarQuantizedVectorsReader.
     * The returned values provide both quantized scoring via scorer() and
     * full-precision rescoring via rescorer().
     */
    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
        return flatVectorsReader.getFloatVectorValues(field);
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
        throw new UnsupportedOperationException("Byte vector search is not supported for Faiss BBQ");
    }

    /**
     * BBQ always uses memory-optimized search — this is not gated by the index-level
     * memory_optimized_search setting. A null target triggers warmup initialization.
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
                "Faiss BBQ requires memory optimized search but searcher could not be loaded for field [" + field + "]"
            );
        }

        memoryOptimizedSearcher.search(target, knnCollector, acceptDocs);
    }

    @Override
    public void search(String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        throw new UnsupportedOperationException("Byte vector search is not supported for Faiss BBQ");
    }

    /**
     * No NativeMemoryCacheManager invalidation or quantization state cache cleanup needed —
     * BBQ doesn't use the k-NN quantization framework or the native memory cache.
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

    /**
     * Double-checked locking for lazy searcher initialization.
     * Once set, the searcher is immutable for the lifetime of this reader.
     */
    private VectorSearcher loadMemoryOptimizedSearcherIfRequired(FieldInfo fieldInfo) {
        if (vectorSearcherHolder.isSet()) {
            return vectorSearcherHolder.getVectorSearcher();
        }

        synchronized (vectorSearcherHolderLockObject) {
            if (vectorSearcherHolder.isSet()) {
                return vectorSearcherHolder.getVectorSearcher();
            }
            final IOSupplier<VectorSearcher> searcherSupplier = getVectorSearcherSupplier(fieldInfo);
            if (searcherSupplier != null) {
                try {
                    vectorSearcherHolder.setVectorSearcher(searcherSupplier.get());
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            } else {
                log.error("Failed to load memory optimized searcher for field [{}]", fieldInfo.getName());
            }
            return vectorSearcherHolder.getVectorSearcher();
        }
    }

    private IOSupplier<VectorSearcher> getVectorSearcherSupplier(FieldInfo fieldInfo) {
        final Map<String, String> attributes = fieldInfo.attributes();
        if (attributes == null || attributes.containsKey(KNN_FIELD) == false) {
            return null;
        }
        final KNNEngine knnEngine = FieldInfoExtractor.extractKNNEngine(fieldInfo);
        if (knnEngine == null) {
            return null;
        }
        final VectorSearcherFactory searcherFactory = knnEngine.getVectorSearcherFactory();
        if (searcherFactory == null) {
            return null;
        }
        final String fileName = KNNCodecUtil.getNativeEngineFileFromFieldInfo(fieldInfo, segmentReadState.segmentInfo);
        if (fileName != null) {
            return () -> searcherFactory.createVectorSearcher(segmentReadState.directory, fileName, fieldInfo, ioContext);
        }
        return null;
    }
}

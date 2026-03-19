/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.Getter;
import lombok.NonNull;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.FileDataHint;
import org.apache.lucene.store.FileTypeHint;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.IOSupplier;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;
import org.opensearch.knn.memoryoptsearch.VectorSearcherFactory;

import org.apache.lucene.util.IOUtils;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.index.mapper.KNNVectorFieldMapper.KNN_FIELD;

/**
 * Abstract base class for KNN vector readers that support memory-optimized search.
 * Provides shared infrastructure for lazy-loading a {@link VectorSearcher} in a thread-safe manner.
 */
@Log4j2
public abstract class AbstractNativeEnginesKnnVectorsReader extends KnnVectorsReader {

    protected final FlatVectorsReader flatVectorsReader;
    protected final SegmentReadState segmentReadState;
    protected final IOContext ioContext;
    protected volatile VectorSearcherHolder vectorSearcherHolder;
    // This lock object ensure that only one thread can initialize vectorSearcherHolder object.
    // This is needed since we are mappings graphs to memory for memory optimized search lazily. But once we make it eager
    // the lock object will not be needed
    protected final Object vectorSearcherHolderLockObject;

    protected AbstractNativeEnginesKnnVectorsReader(final SegmentReadState state, final FlatVectorsReader flatVectorsReader) {
        this.flatVectorsReader = flatVectorsReader;
        this.segmentReadState = state;
        this.ioContext = state.context.withHints(FileTypeHint.DATA, FileDataHint.KNN_VECTORS, DataAccessHint.RANDOM);
        this.vectorSearcherHolder = new VectorSearcherHolder();
        this.vectorSearcherHolderLockObject = new Object();
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
        final List<Closeable> closeables = new ArrayList<>();
        // Close reader.
        closeables.add(flatVectorsReader);

        // Close Vector Searcher
        if (vectorSearcherHolder != null) {
            // We don't need to check if VectorSearcher is null or not because during close IoUtils checks it
            closeables.add(vectorSearcherHolder.getVectorSearcher());
        }
        IOUtils.close(closeables);
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

    protected VectorSearcher loadMemoryOptimizedSearcherIfRequired(final FieldInfo fieldInfo) {
        if (vectorSearcherHolder.isSet()) {
            return vectorSearcherHolder.getVectorSearcher();
        }

        synchronized (vectorSearcherHolderLockObject) {
            if (vectorSearcherHolder.isSet()) {
                return vectorSearcherHolder.getVectorSearcher();
            }
            final IOSupplier<VectorSearcher> searcherSupplier = getVectorSearcherSupplier(fieldInfo);
            // It's supported. There can be a case where a certain index type underlying is not yet supported while
            // KNNEngine itself supports memory optimized searching.
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

    protected IOSupplier<VectorSearcher> getVectorSearcherSupplier(final FieldInfo fieldInfo) {
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
            return () -> searcherFactory.createVectorSearcher(
                segmentReadState.directory,
                fileName,
                fieldInfo,
                ioContext,
                flatVectorsReader
            );
        }

        // Not supported
        return null;
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

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import com.google.common.annotations.VisibleForTesting;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Dedicated format for Faiss SQ vector fields.
 *
 * <p>Uses {@link KNN1040ScalarQuantizedVectorsFormat} for flat vector storage (.vec/.veq files),
 * while HNSW graph construction is delegated to the native Faiss engine (.faiss files). The
 * {@link ScalarEncoding} determines the document bit width (1, 2, or 4) and is resolved from the
 * field's configured bits via {@link ScalarEncodingResolver}; it defaults to 1-bit
 * ({@link ScalarEncoding#SINGLE_BIT_QUERY_NIBBLE}) for backward compatibility.
 *
 * <p>A field is routed to this format when its method parameters contain an {@code sq} encoder
 * with a supported bit width. See {@code FaissCodecFormatResolver} for the routing logic.
 *
 * @see Faiss1040ScalarQuantizedKnnVectorsWriter
 * @see Faiss1040ScalarQuantizedKnnVectorsReader
 */
@Log4j2
public class Faiss1040ScalarQuantizedKnnVectorsFormat extends KnnVectorsFormat {

    private static final String FORMAT_NAME = "Faiss1040ScalarQuantizedKnnVectorsFormat";

    private static final ScalarEncoding DEFAULT_ENCODING = ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;

    // KNN1040ScalarQuantizedVectorsFormat is stateless per encoding, so we cache one instance per
    // encoding and share it across all format instances.
    private static final Map<ScalarEncoding, KNN1040ScalarQuantizedVectorsFormat> FLAT_FORMAT_CACHE = new ConcurrentHashMap<>();

    private final KNN1040ScalarQuantizedVectorsFormat faissSqFlatFormat;
    private final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory;

    /**
     * Returns the flat format for the default (1-bit) encoding. Retained as a static accessor for tests.
     */
    @VisibleForTesting
    static KNN1040ScalarQuantizedVectorsFormat getFaissSqFlatFormat() {
        return flatFormatFor(DEFAULT_ENCODING);
    }

    private static KNN1040ScalarQuantizedVectorsFormat flatFormatFor(final ScalarEncoding encoding) {
        return FLAT_FORMAT_CACHE.computeIfAbsent(encoding, KNN1040ScalarQuantizedVectorsFormat::new);
    }

    public Faiss1040ScalarQuantizedKnnVectorsFormat() {
        this(new NativeIndexBuildStrategyFactory(), DEFAULT_ENCODING);
    }

    public Faiss1040ScalarQuantizedKnnVectorsFormat(final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory) {
        this(nativeIndexBuildStrategyFactory, DEFAULT_ENCODING);
    }

    public Faiss1040ScalarQuantizedKnnVectorsFormat(
        final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory,
        final ScalarEncoding encoding
    ) {
        super(FORMAT_NAME);
        Objects.requireNonNull(encoding, "ScalarEncoding must not be null");
        this.nativeIndexBuildStrategyFactory = nativeIndexBuildStrategyFactory;
        this.faissSqFlatFormat = flatFormatFor(encoding);
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new Faiss1040ScalarQuantizedKnnVectorsWriter(
            state,
            faissSqFlatFormat.fieldsWriter(state),
            faissSqFlatFormat::fieldsReader,
            nativeIndexBuildStrategyFactory
        );
    }

    /**
     * Wraps the Lucene flat vectors reader with {@link Faiss1040ScalarQuantizedFlatVectorsReader} so that
     * the {@link org.apache.lucene.index.FloatVectorValues} returned by the reader implement
     * {@link org.apache.lucene.codecs.lucene95.HasIndexSlice}. This is required because Lucene's
     * HNSW traversal expects all vector values to expose an {@link org.apache.lucene.store.IndexInput},
     * but Lucene's {@code ScalarQuantizedVectorValues} does not implement that interface.
     */
    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new Faiss1040ScalarQuantizedKnnVectorsReader(
            state,
            new Faiss1040ScalarQuantizedFlatVectorsReader(faissSqFlatFormat.fieldsReader(state))
        );
    }

    /**
     * Uses Faiss max dimension since the HNSW graph is built by native Faiss.
     */
    @Override
    public int getMaxDimensions(String fieldName) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.FAISS);
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName() + "(name=" + this.getClass().getSimpleName() + ")";
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import com.google.common.annotations.VisibleForTesting;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;

/**
 * Dedicated format for Faiss SQ vector fields.
 *
 * <p>Uses {@link KNN1040ScalarQuantizedVectorsFormat} with 1-bit quantization
 * ({@link ScalarEncoding#SINGLE_BIT_QUERY_NIBBLE}) for flat vector storage (.vec/.veq files),
 * while HNSW graph construction is delegated to the native Faiss engine (.faiss files).
 *
 * <p>A field is routed to this format when its method parameters contain
 * {@code "encoder": {"name": "sq", "bits": 1}}. See {@code FaissCodecFormatResolver} for the
 * routing logic in {@code BasePerFieldKnnVectorsFormat.getKnnVectorsFormatForField}.
 *
 * @see Faiss1040ScalarQuantizedKnnVectorsWriter
 * @see Faiss1040ScalarQuantizedKnnVectorsReader
 */
@Log4j2
public class Faiss1040ScalarQuantizedKnnVectorsFormat extends KnnVectorsFormat {

    private static final String FORMAT_NAME = "Faiss1040ScalarQuantizedKnnVectorsFormat";

    // Shared across all format instances; KNN1040ScalarQuantizedVectorsFormat is stateless.
    // TODO : We have to make it scalable for other encoding types, not limit this on `ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE`.
    private static final KNN1040ScalarQuantizedVectorsFormat faissSqFlatFormat = new KNN1040ScalarQuantizedVectorsFormat(
        ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE
    );

    private final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory;

    @VisibleForTesting
    static KNN1040ScalarQuantizedVectorsFormat getFaissSqFlatFormat() {
        return faissSqFlatFormat;
    }

    public Faiss1040ScalarQuantizedKnnVectorsFormat() {
        this(new NativeIndexBuildStrategyFactory());
    }

    public Faiss1040ScalarQuantizedKnnVectorsFormat(final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory) {
        super(FORMAT_NAME);
        this.nativeIndexBuildStrategyFactory = nativeIndexBuildStrategyFactory;
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

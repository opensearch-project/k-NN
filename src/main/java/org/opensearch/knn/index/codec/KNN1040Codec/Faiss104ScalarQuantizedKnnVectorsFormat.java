/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;

/**
 * Dedicated format for Faiss BBQ (Better Binary Quantization) vector fields.
 *
 * <p>Uses Lucene's {@link Lucene104ScalarQuantizedVectorsFormat} with 1-bit quantization
 * ({@link ScalarEncoding#SINGLE_BIT_QUERY_NIBBLE}) for flat vector storage (.vec/.veq files),
 * while HNSW graph construction is delegated to the native Faiss engine (.faiss files).
 *
 * <p>A field is routed to this format when its method parameters contain
 * {@code "encoder": {"name": "faiss_bbq"}}. See {@code FaissCodecFormatResolver} for the
 * routing logic in {@code BasePerFieldKnnVectorsFormat.getKnnVectorsFormatForField}.
 *
 * <p>Uses Lucene's max dimension limit since the flat storage is Lucene-managed.
 *
 * @see Faiss104ScalarQuantizedKnnVectorsWriter
 * @see Faiss104ScalarQuantizedKnnVectorsReader
 */
@Log4j2
public class Faiss104ScalarQuantizedKnnVectorsFormat extends KnnVectorsFormat {

    private static final String FORMAT_NAME = "Faiss104ScalarQuantizedKnnVectorsFormat";

    // Shared across all format instances; Lucene104ScalarQuantizedVectorsFormat is stateless.
    private static final Lucene104ScalarQuantizedVectorsFormat bbqFlatFormat = new Lucene104ScalarQuantizedVectorsFormat(
        ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE
    );

    private final int approximateThreshold;

    public Faiss104ScalarQuantizedKnnVectorsFormat() {
        this(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE);
    }

    /**
     * @param approximateThreshold if the number of vectors in a segment is below this threshold,
     *                             HNSW graph building is skipped. A negative value always skips.
     */
    public Faiss104ScalarQuantizedKnnVectorsFormat(int approximateThreshold) {
        super(FORMAT_NAME);
        this.approximateThreshold = approximateThreshold;
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new Faiss104ScalarQuantizedKnnVectorsWriter(state, bbqFlatFormat.fieldsWriter(state), approximateThreshold);
    }

    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new Faiss104ScalarQuantizedKnnVectorsReader(state, bbqFlatFormat.fieldsReader(state));
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
        return this.getClass().getSimpleName()
            + "(name="
            + this.getClass().getSimpleName()
            + ", approximateThreshold="
            + approximateThreshold
            + ")";
    }
}

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
 * @see FaissBBQ1040KnnVectorsWriter
 * @see FaissBBQ1040KnnVectorsReader
 */
@Log4j2
public class FaissBBQ1040KnnVectorsFormat extends KnnVectorsFormat {

    private static final String FORMAT_NAME = "FaissBBQ1040KnnVectorsFormat";

    // Shared across all format instances; Lucene104ScalarQuantizedVectorsFormat is stateless.
    private static final Lucene104ScalarQuantizedVectorsFormat bbqFlatFormat = new Lucene104ScalarQuantizedVectorsFormat(
        ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE
    );

    private final int approximateThreshold;

    public FaissBBQ1040KnnVectorsFormat() {
        this(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE);
    }

    /**
     * @param approximateThreshold if the number of vectors in a segment is below this threshold,
     *                             HNSW graph building is skipped. A negative value always skips.
     */
    public FaissBBQ1040KnnVectorsFormat(int approximateThreshold) {
        super(FORMAT_NAME);
        this.approximateThreshold = approximateThreshold;
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new FaissBBQ1040KnnVectorsWriter(state, bbqFlatFormat.fieldsWriter(state), approximateThreshold);
    }

    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new FaissBBQ1040KnnVectorsReader(state, bbqFlatFormat.fieldsReader(state));
    }

    /**
     * Uses Lucene's max dimension since flat vector storage is Lucene-managed.
     */
    @Override
    public int getMaxDimensions(String fieldName) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE);
    }

    @Override
    public String toString() {
        return "FaissBBQ1040KnnVectorsFormat(name="
            + this.getClass().getSimpleName()
            + ", approximateThreshold="
            + approximateThreshold
            + ")";
    }
}

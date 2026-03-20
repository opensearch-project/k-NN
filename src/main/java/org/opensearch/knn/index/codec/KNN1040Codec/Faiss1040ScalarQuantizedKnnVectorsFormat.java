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
 * {@code "encoder": {"name": "sq", "bits": 1}}. See {@code FaissCodecFormatResolver} for the
 * routing logic in {@code BasePerFieldKnnVectorsFormat.getKnnVectorsFormatForField}.
 *
 * @see Faiss1040ScalarQuantizedKnnVectorsWriter
 * @see Faiss1040ScalarQuantizedKnnVectorsReader
 */
@Log4j2
public class Faiss1040ScalarQuantizedKnnVectorsFormat extends KnnVectorsFormat {

    private static final String FORMAT_NAME = "Faiss1040ScalarQuantizedKnnVectorsFormat";

    // Shared across all format instances; Lucene104ScalarQuantizedVectorsFormat is stateless.
    // TODO : We have to make it scalable for other encoding types, not limit this on `ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE`.
    private static final Lucene104ScalarQuantizedVectorsFormat bbqFlatFormat = new Lucene104ScalarQuantizedVectorsFormat(
        ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE
    );

    public Faiss1040ScalarQuantizedKnnVectorsFormat() {
        super(FORMAT_NAME);
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new Faiss1040ScalarQuantizedKnnVectorsWriter(state, bbqFlatFormat.fieldsWriter(state), bbqFlatFormat::fieldsReader);
    }

    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new Faiss1040ScalarQuantizedKnnVectorsReader(state, bbqFlatFormat.fieldsReader(state));
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

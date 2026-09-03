/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.opensearch.knn.index.codec.scorer.NativeEngines990KnnVectorsScorer;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;
import java.util.Locale;

/**
 * Custom {@link FlatVectorsFormat} implementation to support half-float vectors. This class is mostly identical to
 * {@link org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat}, however we use the custom {@link KNN1040HalfFloatFlatVectorsWriter}
 * and {@link KNN1040HalfFloatFlatVectorsReader} for storage and retrieval of half-float vectors.
 */
public class KNN1040HalfFloatFlatVectorsFormat extends FlatVectorsFormat {

    static final String NAME = "KNN1040HalfFloatFlatVectorsFormat";
    static final String META_CODEC_NAME = "KNN1040HalfFloatFlatVectorsFormatMeta";
    static final String VECTOR_DATA_CODEC_NAME = "KNN1040HalfFloatFlatVectorsFormatData";
    static final String META_EXTENSION = "vemf";
    static final String VECTOR_DATA_EXTENSION = "vec";
    static final int VERSION_START = 0;
    static final int VERSION_CURRENT = VERSION_START;
    static final int DIRECT_MONOTONIC_BLOCK_SHIFT = 16;
    static final int BULK_SCORE_BATCH_SIZE = 64;

    private static final FlatVectorsScorer KNN_1040_HALF_FLOAT_FLAT_VECTORS_SCORER = new PrefetchableFlatVectorScorer(
        new KNN1040HalfFloatVectorScorer(new NativeEngines990KnnVectorsScorer(FlatVectorScorerUtil.getLucene99FlatVectorsScorer()))
    );

    public KNN1040HalfFloatFlatVectorsFormat() {
        super(NAME);
    }

    @Override
    public FlatVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new KNN1040HalfFloatFlatVectorsWriter(state, KNN_1040_HALF_FLOAT_FLAT_VECTORS_SCORER);
    }

    @Override
    public FlatVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new KNN1040HalfFloatFlatVectorsReader(state, KNN_1040_HALF_FLOAT_FLAT_VECTORS_SCORER);
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE);
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "%s(scorer=%s)", getClass().getSimpleName(), KNN_1040_HALF_FLOAT_FLAT_VECTORS_SCORER);
    }
}

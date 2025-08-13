/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec.halffloatcodec;

import java.io.IOException;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

/**
 * A FlatVectorsFormat wrapper that writes half-precision (FP16) vectors.
 */

/**
 * Custom FlatVectorsFormat implementation to support half-float vectors. This class is mostly identical to
 * {@link org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat}, however we use the custom {@link KNN990HalfFloatFlatVectorsWriter}
 * and {@link KNN990HalfFloatFlatVectorsReader} for storage and retrieval of half-float vectors.
 */
public final class KNN990HalfFloatFlatVectorsFormat extends FlatVectorsFormat {

    static final String FORMAT_NAME = "KNN990HalfFloatFlatVectorsFormat";
    static final String META_CODEC_NAME = "Lucene99FlatVectorsFormatMeta";
    static final String VECTOR_DATA_CODEC_NAME = "Lucene99FlatVectorsFormatData";
    static final String META_EXTENSION = "vemf";
    static final String VECTOR_DATA_EXTENSION = "vec";

    static final int VERSION_START = 0;
    static final int VERSION_CURRENT = VERSION_START;

    static final int DIRECT_MONOTONIC_BLOCK_SHIFT = 16;
    private final FlatVectorsScorer vectorsScorer;

    public KNN990HalfFloatFlatVectorsFormat() {
        this(FlatVectorScorerUtil.getLucene99FlatVectorsScorer());
    }

    public KNN990HalfFloatFlatVectorsFormat(FlatVectorsScorer vectorsScorer) {
        super(FORMAT_NAME);
        this.vectorsScorer = vectorsScorer;
    }

    @Override
    public FlatVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new KNN990HalfFloatFlatVectorsWriter(state, vectorsScorer);
    }

    @Override
    public FlatVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new KNN990HalfFloatFlatVectorsReader(state, vectorsScorer);
    }

    @Override
    public String toString() {
        return FORMAT_NAME + "(vectorsScorer=" + vectorsScorer + ")";
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsReader;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

import java.io.IOException;

/**
 * This is a Vector format that will be used for flat vectors that do not create graph data structures.
 */
@Log4j2
public class KNN990FlatVectorsFormat extends KnnVectorsFormat{
    private static Lucene99FlatVectorsFormat lucene99FlatVectorsFormat;
    private static final String FORMAT_NAME = "KNN990FlatVectorsFormat";

    public KNN990FlatVectorsFormat() {
        this(new Lucene99FlatVectorsFormat(new DefaultFlatVectorScorer()));
    }

    public KNN990FlatVectorsFormat(Lucene99FlatVectorsFormat flatVectorsFormat) {
        super(FORMAT_NAME);
        KNN990FlatVectorsFormat.lucene99FlatVectorsFormat = flatVectorsFormat;
    }

    /**
     * Returns a {@link KnnVectorsWriter} to write the vectors to the index.
     *
     * @param state {@link SegmentWriteState}
     */
    @Override
    public KnnVectorsWriter fieldsWriter(final SegmentWriteState state) throws IOException {
        return new Lucene99FlatVectorsWriter(state, FlatVectorScorerUtil.getLucene99FlatVectorsScorer());
    }

    /**
     * Returns a {@link KnnVectorsReader} to read the vectors from the index.
     *
     * @param state {@link SegmentReadState}
     */
    @Override
    public KnnVectorsReader fieldsReader(final SegmentReadState state) throws IOException {
        return new Lucene99FlatVectorsReader(state, FlatVectorScorerUtil.getLucene99FlatVectorsScorer());
    }

    @Override
    public String toString() {
        return "KNN990FlatVectorsFormat(name=KNN990FlatVectorsFormat, flatVectorsFormat=" + lucene99FlatVectorsFormat + ")";
    }

    @Override
    public int getMaxDimensions(String s) { return KnnVectorsFormat.DEFAULT_MAX_DIMENSIONS; }
}

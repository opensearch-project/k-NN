/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.hnsw.*;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsReader;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import java.io.IOException;

/**
 * This is a KNN Vector format that will be used only for flat vectors, when a field is set to index = false.
 * This format is necessary for making sure that graph files are not created when not needed, and is a wrapper around the
 * Lucene99FlatVectorsFormat since that class is not included in the "org.apache.lucene.codecs.KnnVectorsFormat" resources file.
 */
@Log4j2
public class KNN990FlatVectorsFormat extends FlatVectorsFormat {
    private static volatile FlatVectorsFormat FLAT_VECTORS_FORMAT;
    private static final String FORMAT_NAME = "KNN990FlatVectorsFormat";

    public KNN990FlatVectorsFormat() {
        this(new Lucene99FlatVectorsFormat(FlatVectorScorerUtil.getLucene99FlatVectorsScorer()));
    }

    public KNN990FlatVectorsFormat(final FlatVectorsFormat flatVectorsFormat) {
        super(FORMAT_NAME);
        KNN990FlatVectorsFormat.FLAT_VECTORS_FORMAT = flatVectorsFormat;
    }

    /**
     * Returns a {@link FlatVectorsWriter} to write the vectors to the index.
     *
     * @param state {@link SegmentWriteState}
     */
    @Override
    public FlatVectorsWriter fieldsWriter(final SegmentWriteState state) throws IOException {
        return new Lucene99FlatVectorsWriter(state, FlatVectorScorerUtil.getLucene99FlatVectorsScorer());
    }

    /**
     * Returns a {@link FlatVectorsReader} to read the vectors from the index.
     *
     * @param state {@link SegmentReadState}
     */
    @Override
    public FlatVectorsReader fieldsReader(final SegmentReadState state) throws IOException {
        return new Lucene99FlatVectorsReader(state, FlatVectorScorerUtil.getLucene99FlatVectorsScorer());
    }

    @Override
    public String toString() {
        return "KNN990FlatVectorsFormat(name=" + this.getClass().getSimpleName() + ", flatVectorsFormat=" + FLAT_VECTORS_FORMAT + ")";
    }
}

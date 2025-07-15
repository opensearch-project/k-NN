/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsReader;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import java.io.IOException;

/**
 * This is a Vector format that will be used only for flat vectors, when a field is set to index = false.
 */
@Log4j2
public class KNN990FlatVectorsFormat extends FlatVectorsFormat {
    private static FlatVectorsFormat flatVectorsFormat;
    private static final String FORMAT_NAME = "KNN990FlatVectorsFormat";

    public KNN990FlatVectorsFormat() {
        this(new Lucene99FlatVectorsFormat(new DefaultFlatVectorScorer()));
    }

    public KNN990FlatVectorsFormat(final FlatVectorsFormat flatVectorsFormat) {
        super(FORMAT_NAME);
        KNN990FlatVectorsFormat.flatVectorsFormat = flatVectorsFormat;
    }

    /**
     * Returns a {@link FlatVectorsWriter} to write the vectors to the index.
     *
     * @param state {@link SegmentWriteState}
     */
    @Override
    public FlatVectorsWriter fieldsWriter(final SegmentWriteState state) throws IOException {
        return new Lucene99FlatVectorsWriter(state, new DefaultFlatVectorScorer());
    }

    /**
     * Returns a {@link FlatVectorsReader} to read the vectors from the index.
     *
     * @param state {@link SegmentReadState}
     */
    @Override
    public FlatVectorsReader fieldsReader(final SegmentReadState state) throws IOException {
        return new Lucene99FlatVectorsReader(state, new DefaultFlatVectorScorer());
    }

    @Override
    public String toString() {
        return "KNN990FlatVectorsFormat(name=" + this.getClass().getSimpleName() + ", flatVectorsFormat=" + flatVectorsFormat + ")";
    }
}

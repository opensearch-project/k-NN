/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorScorer;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsReader;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

import java.io.IOException;

/**
 * A {@link Lucene104ScalarQuantizedVectorsFormat} that uses a {@link Faiss104ScalarQuantizedVectorScorer}
 * to take advantage of SIMD-accelerated scoring during search.
 */
public class KNN104ScalarQuantizedVectorsFormat extends Lucene104ScalarQuantizedVectorsFormat {

    private static final Lucene104ScalarQuantizedVectorScorer PREFETCHING_SCORER = new Faiss104ScalarQuantizedVectorScorer(
        FlatVectorScorerUtil.getLucene99FlatVectorsScorer()
    );

    private static final Lucene99FlatVectorsFormat RAW_VECTOR_FORMAT = new Lucene99FlatVectorsFormat(
        FlatVectorScorerUtil.getLucene99FlatVectorsScorer()
    );

    private final ScalarEncoding encoding;

    public KNN104ScalarQuantizedVectorsFormat(final ScalarEncoding encoding) {
        super(encoding);
        this.encoding = encoding;
    }

    @Override
    public FlatVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new Lucene104ScalarQuantizedVectorsWriter(state, encoding, RAW_VECTOR_FORMAT.fieldsWriter(state), PREFETCHING_SCORER);
    }

    @Override
    public String toString() {
        return String.format(
            "%s(encoding=%s, scorer=%s, rawVectorFormat=%s)",
            getClass().getSimpleName(),
            encoding,
            PREFETCHING_SCORER,
            RAW_VECTOR_FORMAT
        );
    }

    @Override
    public FlatVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new Lucene104ScalarQuantizedVectorsReader(state, RAW_VECTOR_FORMAT.fieldsReader(state), PREFETCHING_SCORER);
    }
}

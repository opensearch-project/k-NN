/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;
import org.opensearch.knn.index.VectorDataType;

import java.util.concurrent.ExecutorService;

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_BEAM_WIDTH;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_MAX_CONN;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_NUM_MERGE_WORKER;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.HNSW_GRAPH_THRESHOLD;

/**
 * HNSW + SQ 1-bit format for {@code half_float} with the same mechanism as
 * {@link KNN1040HnswScalarQuantizedVectorsFormat}, but with an FP16 raw delegate instead of FP32.
 * Separate class on purpose as Lucene reconstructs formats by name via a no-arg constructor, so
 * sharing a name with the FLOAT variant would silently pick the wrong delegate on read.
 */
public class KNN1040HnswHalfFloatScalarQuantizedVectorsFormat extends KNN1040HnswScalarQuantizedVectorsFormat {

    public KNN1040HnswHalfFloatScalarQuantizedVectorsFormat() {
        this(ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE, DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH, DEFAULT_NUM_MERGE_WORKER, null);
    }

    public KNN1040HnswHalfFloatScalarQuantizedVectorsFormat(
        ScalarEncoding encoding,
        int maxConn,
        int beamWidth,
        int numMergeWorkers,
        ExecutorService mergeExec
    ) {
        this(encoding, maxConn, beamWidth, numMergeWorkers, mergeExec, HNSW_GRAPH_THRESHOLD);
    }

    public KNN1040HnswHalfFloatScalarQuantizedVectorsFormat(
        ScalarEncoding encoding,
        int maxConn,
        int beamWidth,
        int numMergeWorkers,
        ExecutorService mergeExec,
        int tinySegmentsThreshold
    ) {
        super(encoding, maxConn, beamWidth, numMergeWorkers, mergeExec, tinySegmentsThreshold, VectorDataType.HALF_FLOAT);
    }
}

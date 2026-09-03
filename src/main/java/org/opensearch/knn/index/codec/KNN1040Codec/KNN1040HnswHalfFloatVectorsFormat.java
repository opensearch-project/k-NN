/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.TaskExecutor;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;
import java.util.Locale;
import java.util.concurrent.ExecutorService;

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_BEAM_WIDTH;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_MAX_CONN;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_NUM_MERGE_WORKER;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.HNSW_GRAPH_THRESHOLD;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.MAXIMUM_BEAM_WIDTH;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.MAXIMUM_MAX_CONN;

/**
 * HNSW format for half-float vectors. Mostly identical to {@link org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat},
 * but uses {@link KNN1040HalfFloatFlatVectorsFormat} as the flat vector delegate so the graph's backing
 * storage is FP16 instead of FP32, inheriting its SIMD-accelerated scorer for graph traversal.
 */
public class KNN1040HnswHalfFloatVectorsFormat extends KnnVectorsFormat {

    private final int maxConn;
    private final int beamWidth;
    private final int tinySegmentsThreshold;
    private final int numMergeWorkers;
    private final TaskExecutor mergeExec;
    private static final FlatVectorsFormat flatVectorsFormat = new KNN1040HalfFloatFlatVectorsFormat();

    private static final String NAME = "KNN1040HnswHalfFloatVectorsFormat";

    public KNN1040HnswHalfFloatVectorsFormat() {
        this(DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH, DEFAULT_NUM_MERGE_WORKER, null, HNSW_GRAPH_THRESHOLD);
    }

    public KNN1040HnswHalfFloatVectorsFormat(int maxConn, int beamWidth, int numMergeWorkers, ExecutorService mergeExec) {
        this(maxConn, beamWidth, numMergeWorkers, mergeExec, HNSW_GRAPH_THRESHOLD);
    }

    public KNN1040HnswHalfFloatVectorsFormat(
        int maxConn,
        int beamWidth,
        int numMergeWorkers,
        ExecutorService mergeExec,
        int tinySegmentsThreshold
    ) {
        super(NAME);
        if (maxConn <= 0 || maxConn > MAXIMUM_MAX_CONN) {
            throw new IllegalArgumentException(
                "maxConn must be positive and less than or equal to " + MAXIMUM_MAX_CONN + "; maxConn=" + maxConn
            );
        }
        if (beamWidth <= 0 || beamWidth > MAXIMUM_BEAM_WIDTH) {
            throw new IllegalArgumentException(
                "beamWidth must be positive and less than or equal to " + MAXIMUM_BEAM_WIDTH + "; beamWidth=" + beamWidth
            );
        }
        if (numMergeWorkers == 1 && mergeExec != null) {
            throw new IllegalArgumentException("No executor service is needed as we'll use single thread to merge");
        }
        this.maxConn = maxConn;
        this.beamWidth = beamWidth;
        this.tinySegmentsThreshold = tinySegmentsThreshold;
        this.numMergeWorkers = numMergeWorkers;
        this.mergeExec = mergeExec != null ? new TaskExecutor(mergeExec) : null;
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new Lucene99HnswVectorsWriter(
            state,
            maxConn,
            beamWidth,
            flatVectorsFormat,
            flatVectorsFormat.fieldsWriter(state),
            numMergeWorkers,
            mergeExec,
            tinySegmentsThreshold
        );
    }

    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new Lucene99HnswVectorsReader(state, flatVectorsFormat.fieldsReader(state));
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE);
    }

    @Override
    public String toString() {
        return String.format(
            Locale.ROOT,
            "%s(maxConn=%d, beamWidth=%d, tinySegmentsThreshold=%d, flatVectorFormat=%s)",
            getClass().getSimpleName(),
            maxConn,
            beamWidth,
            tinySegmentsThreshold,
            flatVectorsFormat
        );
    }
}

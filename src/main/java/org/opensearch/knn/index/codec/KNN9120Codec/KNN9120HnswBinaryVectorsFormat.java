/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.TaskExecutor;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;
import java.util.concurrent.ExecutorService;

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_BEAM_WIDTH;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_MAX_CONN;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_NUM_MERGE_WORKER;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.MAXIMUM_BEAM_WIDTH;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.MAXIMUM_MAX_CONN;
import static org.opensearch.knn.index.engine.KNNEngine.getMaxDimensionByEngine;

/**
 * Custom KnnVectorsFormat implementation to support binary vectors. This class is mostly identical to
 * {@link org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat}, however we use the custom {@link KNN9120BinaryVectorScorer}
 * to perform hamming bit scoring.
 */
public final class KNN9120HnswBinaryVectorsFormat extends KnnVectorsFormat {

    private final int maxConn;
    private final int beamWidth;
    private static final FlatVectorsFormat flatVectorsFormat = new Lucene99FlatVectorsFormat(new KNN9120BinaryVectorScorer());
    private final int numMergeWorkers;
    private final TaskExecutor mergeExec;

    private static final String NAME = "KNN990HnswBinaryVectorsFormat";

    /**
     * Constructor logic is identical to {@link org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat#Lucene99HnswVectorsFormat()}
     */
    public KNN9120HnswBinaryVectorsFormat() {
        this(DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH, DEFAULT_NUM_MERGE_WORKER, null);
    }

    /**
     * Constructor logic is identical to {@link org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat#Lucene99HnswVectorsFormat(int, int)}
     */
    public KNN9120HnswBinaryVectorsFormat(int maxConn, int beamWidth) {
        this(maxConn, beamWidth, 1, null);
    }

    /**
     * Constructor logic is identical to {@link org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat#Lucene99HnswVectorsFormat(int, int, int, java.util.concurrent.ExecutorService)}
     */
    public KNN9120HnswBinaryVectorsFormat(int maxConn, int beamWidth, int numMergeWorkers, ExecutorService mergeExec) {
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
        this.maxConn = maxConn;
        this.beamWidth = beamWidth;
        if (numMergeWorkers == 1 && mergeExec != null) {
            throw new IllegalArgumentException("No executor service is needed as we'll use single thread to merge");
        }
        this.numMergeWorkers = numMergeWorkers;
        if (mergeExec != null) {
            this.mergeExec = new TaskExecutor(mergeExec);
        } else {
            this.mergeExec = null;
        }
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new Lucene99HnswVectorsWriter(
            state,
            this.maxConn,
            this.beamWidth,
            flatVectorsFormat.fieldsWriter(state),
            this.numMergeWorkers,
            this.mergeExec
        );
    }

    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new Lucene99HnswVectorsReader(state, flatVectorsFormat.fieldsReader(state));
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return getMaxDimensionByEngine(KNNEngine.LUCENE);
    }

    @Override
    public String toString() {
        return "KNN990HnswBinaryVectorsFormat(name=KNN990HnswBinaryVectorsFormat, maxConn="
            + this.maxConn
            + ", beamWidth="
            + this.beamWidth
            + ", flatVectorFormat="
            + flatVectorsFormat
            + ")";
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

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

import java.io.IOException;
import java.util.concurrent.ExecutorService;

public class KNN990HnswBinaryVectorsFormat extends KnnVectorsFormat {

    private final int maxConn;
    private final int beamWidth;
    private static final FlatVectorsFormat flatVectorsFormat = new Lucene99FlatVectorsFormat(new KNN990BinaryVectorScorer());
    private final int numMergeWorkers;
    private final TaskExecutor mergeExec;

    private static final String NAME = "KNN990HnswBinaryVectorsFormat";

    public KNN990HnswBinaryVectorsFormat() {
        this(16, 100, 1, (ExecutorService) null);
    }

    public KNN990HnswBinaryVectorsFormat(int maxConn, int beamWidth) {
        this(maxConn, beamWidth, 1, (ExecutorService) null);
    }

    public KNN990HnswBinaryVectorsFormat(int maxConn, int beamWidth, int numMergeWorkers, ExecutorService mergeExec) {
        super(NAME);
        if (maxConn > 0 && maxConn <= 512) {
            if (beamWidth > 0 && beamWidth <= 3200) {
                this.maxConn = maxConn;
                this.beamWidth = beamWidth;
                if (numMergeWorkers == 1 && mergeExec != null) {
                    throw new IllegalArgumentException("No executor service is needed as we'll use single thread to merge");
                } else {
                    this.numMergeWorkers = numMergeWorkers;
                    if (mergeExec != null) {
                        this.mergeExec = new TaskExecutor(mergeExec);
                    } else {
                        this.mergeExec = null;
                    }

                }
            } else {
                throw new IllegalArgumentException("beamWidth must be positive and less than or equal to 3200; beamWidth=" + beamWidth);
            }
        } else {
            throw new IllegalArgumentException("maxConn must be positive and less than or equal to 512; maxConn=" + maxConn);
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
        return 1024;
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

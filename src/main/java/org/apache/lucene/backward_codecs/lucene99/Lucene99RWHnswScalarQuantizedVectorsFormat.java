/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.apache.lucene.backward_codecs.lucene99;

import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsWriter;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.TaskExecutor;

import java.io.IOException;
import java.util.concurrent.ExecutorService;

/**
 * A read-write implementation of {@link Lucene99HnswScalarQuantizedVectorsFormat}.
 *
 * <p>This format extends the base Lucene99HnswScalarQuantizedVectorsFormat to provide write
 * capabilities for the older format. Lucene stopped writing to this format, and the new format
 * doesn't accept confidenceInterval and compress flags as input parameters.
 *
 * <p>This implementation allows writing vectors with scalar quantization using the legacy format
 * that supports these configuration options.
 *
 * @see <a href="https://github.com/apache/lucene/pull/15223">Lucene PR #15223</a>
 */
public class Lucene99RWHnswScalarQuantizedVectorsFormat extends Lucene99HnswScalarQuantizedVectorsFormat {

    private final FlatVectorsFormat flatVectorsFormat;
    private final int maxConn;
    private final int beamWidth;
    private final int numMergeThreads;
    private final TaskExecutor executorService;

    /**
     * Constructs a new format with the specified HNSW and scalar quantization parameters.
     *
     * @param maxConn            the maximum number of connections per node in the HNSW graph
     * @param beamWidth          the size of the queue maintained during graph construction
     * @param numMergeThreads    the number of threads to use during segment merges
     * @param bits               the number of bits for scalar quantization (typically 7 or 8)
     * @param compressFlag       whether to compress the quantized vectors
     * @param confidenceInterval the confidence interval for quantization (0.0 to 1.0)
     * @param executor           the executor service for parallel operations, or null for single-threaded
     */
    public Lucene99RWHnswScalarQuantizedVectorsFormat(
        int maxConn,
        int beamWidth,
        int numMergeThreads,
        int bits,
        boolean compressFlag,
        Float confidenceInterval,
        ExecutorService executor
    ) {
        super(maxConn, beamWidth, numMergeThreads, bits, compressFlag, confidenceInterval, executor);
        this.flatVectorsFormat = new Lucene99RWScalarQuantizedVectorsFormat(confidenceInterval, bits, compressFlag);
        this.maxConn = maxConn;
        this.beamWidth = beamWidth;
        this.numMergeThreads = numMergeThreads;
        if (executor != null) {
            this.executorService = new TaskExecutor(executor);
        } else {
            this.executorService = null;
        }
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new Lucene99HnswVectorsWriter(
            state,
            maxConn,
            beamWidth,
            flatVectorsFormat.fieldsWriter(state),
            numMergeThreads,
            executorService
        );
    }
}

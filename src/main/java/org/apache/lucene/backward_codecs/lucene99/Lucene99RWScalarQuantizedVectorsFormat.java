/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.apache.lucene.backward_codecs.lucene99;

import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.SegmentWriteState;

import java.io.IOException;

/**
 * A read-write implementation of {@link Lucene99ScalarQuantizedVectorsFormat}.
 *
 * <p>This format extends the base Lucene99ScalarQuantizedVectorsFormat to provide write
 * capabilities for the older format. Lucene stopped writing to this format, and the new format
 * doesn't accept confidenceInterval and compress flags as input parameters.
 *
 * <p>This implementation allows writing flat vectors with scalar quantization using the legacy
 * format that supports these configuration options.
 *
 * @see <a href="https://github.com/apache/lucene/pull/15223">Lucene PR #15223</a>
 */
public class Lucene99RWScalarQuantizedVectorsFormat extends Lucene99ScalarQuantizedVectorsFormat {

    private final Float confidenceInterval;
    private final byte bits;
    private final boolean compress;
    // This is main scorer that used during indexing and search.
    private final static FlatVectorsScorer flatVectorScorer = FlatVectorScorerUtil.getLucene99ScalarQuantizedVectorsScorer();
    // Scorer passed to this format is not used for scoring.
    private static final FlatVectorsFormat rawVectorFormat = new Lucene99FlatVectorsFormat(flatVectorScorer);

    /**
     * Constructs a new format with the specified scalar quantization parameters.
     *
     * @param confidenceInterval the confidence interval for quantization (0.0 to 1.0), or null for default
     * @param bits the number of bits for scalar quantization (typically 7 or 8)
     * @param compress whether to compress the quantized vectors
     */
    public Lucene99RWScalarQuantizedVectorsFormat(Float confidenceInterval, int bits, boolean compress) {
        super(confidenceInterval, bits, compress);
        this.confidenceInterval = confidenceInterval;
        this.bits = (byte) bits;
        this.compress = compress;
    }

    @Override
    public FlatVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new Lucene99ScalarQuantizedVectorsWriter(
            state,
            this.confidenceInterval,
            this.bits,
            this.compress,
            rawVectorFormat.fieldsWriter(state),
            flatVectorScorer
        );
    }
}

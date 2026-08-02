/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.VectorScorer;

import java.io.IOException;

/**
 * A {@link FloatVectorValues} wrapper that holds both the full-precision float delegate (backed by
 * the {@code .vec} file) and the underlying quantized byte delegate (backed by the {@code .veq} file).
 *
 * <p>The wrapper exists so callers can reach either representation without reflection: the search
 * scorer needs the quantized codes; warmup needs the full-precision bytes. Consumers pick via
 * {@link #getFloatVectorValues()} or {@link #getQuantizedVectorValues()} rather than relying on a
 * single ambiguous slice.
 *
 * <p>Historically this class implemented {@code HasIndexSlice#getSlice()}. It was removed because
 * the delegating {@code vectorValue(ord)} reads full-precision floats from {@code .vec} while the
 * quantized values expose the {@code .veq} slice — a generic prefetch caller trusting
 * {@code HasIndexSlice} would warm the wrong file, defeating the fetch-phase prefetch entirely.
 *
 * <p>For an empty vector segment, the quantized delegate may be {@code null}.
 */
@Getter
@RequiredArgsConstructor
class ScalarQuantizedFloatVectorValues extends FloatVectorValues {
    /**
     * The full-precision float delegate (reads the {@code .vec} file).
     */
    private final FloatVectorValues floatVectorValues;
    /**
     * The quantized byte delegate (reads the {@code .veq} file), or {@code null} for empty
     * segments where Lucene does not expose one.
     */
    private final QuantizedByteVectorValues quantizedVectorValues;

    @Override
    public int dimension() {
        return floatVectorValues.dimension();
    }

    @Override
    public int size() {
        return floatVectorValues.size();
    }

    @Override
    public float[] vectorValue(int ord) throws IOException {
        return floatVectorValues.vectorValue(ord);
    }

    @Override
    public FloatVectorValues copy() throws IOException {
        return new ScalarQuantizedFloatVectorValues(
            floatVectorValues.copy(),
            quantizedVectorValues == null ? null : quantizedVectorValues.copy()
        );
    }

    @Override
    public VectorEncoding getEncoding() {
        return floatVectorValues.getEncoding();
    }

    @Override
    public DocIndexIterator iterator() {
        return floatVectorValues.iterator();
    }

    /**
     * Returns a {@link VectorScorer} that scores the query against the quantized codes in
     * {@code .veq}. Scoring is the quantized-vs-quantized path, not the full-precision one — the
     * {@code .vec} floats are reserved for rescore/fetch via {@link #rescorer(float[])}.
     */
    @Override
    public VectorScorer scorer(float[] target) throws IOException {
        return quantizedVectorValues == null ? null : quantizedVectorValues.scorer(target);
    }

    /**
     * Returns a {@link VectorScorer} for rescoring candidates against the given target vector
     * using full-precision vectors. Delegates to the underlying {@link FloatVectorValues}.
     *
     * @param target the query vector to score against
     * @return a {@link VectorScorer} for exact rescoring, or {@code null} if not supported
     * @throws IOException if an I/O error occurs
     */
    @Override
    public VectorScorer rescorer(final float[] target) throws IOException {
        return floatVectorValues.rescorer(target);
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;

import java.io.IOException;

/**
 * An empty implementation of {@link QuantizedByteVectorValues} used when Lucene returns empty
 * vector values that do not expose the {@code quantizedVectorValues} field.
 *
 * <p>This class delegates size and iteration operations to the underlying {@link FloatVectorValues}
 * while returning empty values for quantization-specific operations. It maintains the
 * {@link ScalarQuantizedFloatVectorValues} return type expected by callers without fabricating
 * quantized data for a segment that contains no vectors.
 */
class EmptyQuantizedByteVectorValues extends QuantizedByteVectorValues {
    private final FloatVectorValues floatVectorValues;
    private final byte[] emptyVector = new byte[0];

    EmptyQuantizedByteVectorValues(final FloatVectorValues floatVectorValues) {
        this.floatVectorValues = floatVectorValues;
    }

    @Override
    public int dimension() {
        return floatVectorValues.dimension();
    }

    @Override
    public int size() {
        return floatVectorValues.size();
    }

    @Override
    public byte[] vectorValue(int ord) throws IOException {
        return emptyVector;
    }

    @Override
    public DocIndexIterator iterator() {
        return floatVectorValues.iterator();
    }

    @Override
    public EmptyQuantizedByteVectorValues copy() throws IOException {
        return new EmptyQuantizedByteVectorValues(floatVectorValues.copy());
    }

    @Override
    public VectorEncoding getEncoding() {
        return VectorEncoding.BYTE;
    }

    @Override
    public VectorScorer scorer(float[] target) throws IOException {
        return null;
    }

    @Override
    public IndexInput getSlice() {
        return null;
    }

    @Override
    public OptimizedScalarQuantizer.QuantizationResult getCorrectiveTerms(int vectorOrd) throws IOException {
        return null;
    }

    @Override
    public OptimizedScalarQuantizer getQuantizer() {
        return null;
    }

    @Override
    public ScalarEncoding getScalarEncoding() {
        return null;
    }

    @Override
    public float[] getCentroid() throws IOException {
        return null;
    }

    @Override
    public float getCentroidDP() throws IOException {
        return 0f;
    }
}

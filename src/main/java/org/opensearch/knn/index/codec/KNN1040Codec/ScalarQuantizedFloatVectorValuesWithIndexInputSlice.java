/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.RequiredArgsConstructor;
import org.apache.lucene.codecs.lucene104.QuantizedByteVectorValues;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;

/**
 * A {@link FloatVectorValues} wrapper that implements {@link HasIndexSlice} to enable I/O prefetching
 * during HNSW graph traversal.
 *
 * <p>Lucene's {@code ScalarQuantizedVectorValues} (returned by
 * {@code Lucene104ScalarQuantizedVectorsReader.getFloatVectorValues()}) does not implement
 * {@link HasIndexSlice}. Lucene's prefetch-enabled flat vector scorer
 * ({@link org.apache.lucene.codecs.hnsw.FlatVectorsScorer}) requires this interface to obtain the
 * underlying {@link IndexInput} for issuing prefetch hints ahead of scoring.
 *
 * <p>This class bridges the gap by delegating all {@link FloatVectorValues} operations to the original
 * instance, while implementing {@link HasIndexSlice#getSlice()} by returning the {@link IndexInput}
 * from the underlying {@link QuantizedByteVectorValues}. The quantized byte values hold the on-disk
 * slice that contains the quantized vector data and correction factors used during scoring.
 */
@RequiredArgsConstructor
class ScalarQuantizedFloatVectorValuesWithIndexInputSlice extends FloatVectorValues implements HasIndexSlice {
    private final FloatVectorValues floatVectorValues;
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
        return new ScalarQuantizedFloatVectorValuesWithIndexInputSlice(floatVectorValues.copy(), quantizedVectorValues.copy());
    }

    @Override
    public VectorEncoding getEncoding() {
        return floatVectorValues.getEncoding();
    }

    @Override
    public IndexInput getSlice() {
        return quantizedVectorValues.getSlice();
    }

    @Override
    public DocIndexIterator iterator() {
        return floatVectorValues.iterator();
    }

    @Override
    public VectorScorer scorer(float[] target) throws IOException {
        return floatVectorValues.scorer(target);
    }

    @Override
    public VectorScorer rescorer(final float[] target) throws IOException {
        return floatVectorValues.rescorer(target);
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

import java.io.IOException;

/**
 * A {@link FlatVectorsReader} wrapper for Faiss SQ vector fields that exposes both the
 * full-precision floats and the quantized codes on the {@link FloatVectorValues} returned by
 * {@link #getFloatVectorValues(String)}.
 *
 * <p>Lucene's {@code Lucene104ScalarQuantizedVectorsReader} returns a {@code ScalarQuantizedVectorValues}
 * that hides its quantized byte delegate behind a private field. Callers on the plugin side
 * (warmup, native scoring) need both the full-precision {@code .vec} floats and the quantized
 * {@code .veq} codes, so this reader wraps the delegate's values with
 * {@link ScalarQuantizedFloatVectorValues}, which carries both delegates explicitly.
 *
 * <p>The resulting reader hierarchy is:
 * <pre>
 *   Faiss1040ScalarQuantizedKnnVectorsReader
 *     └─ Faiss1040ScalarQuantizedFlatVectorsReader  (this class)
 *          └─ Lucene104ScalarQuantizedVectorsReader  (delegate)
 * </pre>
 *
 * <p>All other operations are delegated directly to the underlying reader.
 */
public class Faiss1040ScalarQuantizedFlatVectorsReader extends FlatVectorsReader {
    private final FlatVectorsReader delegateFlatVectorsReader;

    /**
     * @param lucene104ScalarQuantizedVectorsReader the delegate reader whose {@link FloatVectorValues}
     *                                              will be wrapped to implement {@code HasIndexSlice}
     */
    protected Faiss1040ScalarQuantizedFlatVectorsReader(final FlatVectorsReader lucene104ScalarQuantizedVectorsReader) {
        super();
        this.delegateFlatVectorsReader = lucene104ScalarQuantizedVectorsReader;
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, float[] target) throws IOException {
        return delegateFlatVectorsReader.getRandomVectorScorer(field, target);
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, byte[] target) throws IOException {
        return delegateFlatVectorsReader.getRandomVectorScorer(field, target);
    }

    @Override
    public void checkIntegrity() throws IOException {
        delegateFlatVectorsReader.checkIntegrity();
    }

    /**
     * Returns {@link FloatVectorValues} wrapped with {@link ScalarQuantizedFloatVectorValues},
     * which exposes both the full-precision float delegate and the quantized byte delegate via
     * dedicated getters. Empty values are wrapped with no quantized backing because Lucene does
     * not expose one.
     */
    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
        final FloatVectorValues floatVectorValues = delegateFlatVectorsReader.getFloatVectorValues(field);
        if (floatVectorValues == null) {
            return null;
        }

        if (floatVectorValues.size() == 0) {
            return new ScalarQuantizedFloatVectorValues(floatVectorValues, null);
        }

        return new ScalarQuantizedFloatVectorValues(
            floatVectorValues,
            KNN1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(floatVectorValues)
        );
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
        return delegateFlatVectorsReader.getByteVectorValues(field);
    }

    @Override
    public void close() throws IOException {
        delegateFlatVectorsReader.close();
    }

    @Override
    public long ramBytesUsed() {
        return delegateFlatVectorsReader.ramBytesUsed();
    }

    @Override
    public FlatVectorsScorer getFlatVectorScorer(String field) throws IOException {
        return delegateFlatVectorsReader.getFlatVectorScorer(field);
    }

}

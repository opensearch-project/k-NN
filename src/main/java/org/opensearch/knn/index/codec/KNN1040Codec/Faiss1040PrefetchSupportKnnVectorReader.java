/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

import java.io.IOException;

/**
 * A {@link FlatVectorsReader} wrapper that adds prefetch support for Faiss SQ (1-bit) vector fields.
 *
 * <p>Lucene's {@code Lucene104ScalarQuantizedVectorsReader} returns a {@code ScalarQuantizedVectorValues}
 * from {@link #getFloatVectorValues(String)}, which does not implement
 * {@link org.apache.lucene.codecs.lucene95.HasIndexSlice}. However, Lucene's prefetch-enabled HNSW
 * graph traversal requires all {@link FloatVectorValues} to implement {@code HasIndexSlice} so that
 * the underlying {@link org.apache.lucene.store.IndexInput} can be accessed for I/O prefetching.
 *
 * <p>This reader solves the problem by wrapping the delegate's {@link FloatVectorValues} with
 * {@link ScalarQuantizedFloatVectorValuesWithIndexInputSlice}, which implements {@code HasIndexSlice}
 * by exposing the {@link org.apache.lucene.store.IndexInput} from the quantized byte vector values.
 *
 * <p>The resulting reader hierarchy is:
 * <pre>
 *   Faiss1040ScalarQuantizedKnnVectorsReader
 *     └─ Faiss1040PrefetchSupportKnnVectorReader  (this class)
 *          └─ Lucene104ScalarQuantizedVectorsReader  (delegate)
 * </pre>
 *
 * <p>All other operations are delegated directly to the underlying reader.
 */
public class Faiss1040PrefetchSupportKnnVectorReader extends FlatVectorsReader {
    private final FlatVectorsReader delegateFlatVectorsReader;

    /**
     * @param lucene104ScalarQuantizedVectorsReader the delegate reader whose {@link FloatVectorValues}
     *                                              will be wrapped to support prefetch via {@code HasIndexSlice}
     */
    protected Faiss1040PrefetchSupportKnnVectorReader(final FlatVectorsReader lucene104ScalarQuantizedVectorsReader) {
        super(lucene104ScalarQuantizedVectorsReader.getFlatVectorScorer());
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
     * Returns {@link FloatVectorValues} wrapped with {@link ScalarQuantizedFloatVectorValuesWithIndexInputSlice}
     * so that the result implements {@link org.apache.lucene.codecs.lucene95.HasIndexSlice}, enabling
     * prefetch during HNSW graph traversal.
     */
    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
        final FloatVectorValues floatVectorValues = delegateFlatVectorsReader.getFloatVectorValues(field);
        return new ScalarQuantizedFloatVectorValuesWithIndexInputSlice(
            floatVectorValues,
            Faiss1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(floatVectorValues)
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
}

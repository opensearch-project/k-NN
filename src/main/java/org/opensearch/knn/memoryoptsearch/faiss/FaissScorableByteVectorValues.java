/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.KnnVectorValues.DocIndexIterator;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

import java.io.IOException;

/**
 * Wraps raw {@link ByteVectorValues} and adds scoring capability via a {@link FlatVectorsScorer}.
 *
 * <p>All iteration/random-access methods delegate to the underlying values unchanged.
 * {@link #copy()} returns a fresh wrapper via {@code delegate.copy()} so that each scorer
 * holds an independent read cursor.
 *
 * <p>An optional {@link DocIndexIterator} can be supplied at construction time.
 * When present, both {@link #iterator()} and the scorer returned by {@link #scorer(byte[])}
 * use this iterator instead of the delegate's default one.
 */
public class FaissScorableByteVectorValues extends ByteVectorValues {

    private final ByteVectorValues delegate;
    private final FlatVectorsScorer flatVectorsScorer;
    private final VectorSimilarityFunction similarityFunction;
    private final DocIndexIterator overrideIterator;

    public FaissScorableByteVectorValues(
        final ByteVectorValues delegate,
        final FlatVectorsScorer flatVectorsScorer,
        final VectorSimilarityFunction similarityFunction,
        final DocIndexIterator overrideIterator
    ) {
        if (delegate == null) {
            throw new IllegalArgumentException("delegate must not be null");
        }
        this.delegate = delegate;
        this.flatVectorsScorer = flatVectorsScorer;
        this.similarityFunction = similarityFunction;
        this.overrideIterator = overrideIterator;
    }

    // ---- RandomAccessVectorValues ----

    @Override
    public int dimension() {
        return delegate.dimension();
    }

    @Override
    public int size() {
        return delegate.size();
    }

    @Override
    public byte[] vectorValue(int ord) throws IOException {
        return delegate.vectorValue(ord);
    }

    @Override
    public int ordToDoc(int ord) {
        return delegate.ordToDoc(ord);
    }

    @Override
    public Bits getAcceptOrds(Bits acceptDocs) {
        return delegate.getAcceptOrds(acceptDocs);
    }

    @Override
    public FaissScorableByteVectorValues copy() throws IOException {
        return new FaissScorableByteVectorValues(delegate.copy(), flatVectorsScorer, similarityFunction, overrideIterator);
    }

    @Override
    public DocIndexIterator iterator() {
        if (overrideIterator != null) {
            return overrideIterator;
        }
        return delegate.iterator();
    }

    // ---- Scorer ----

    /**
     * Returns a {@link VectorScorer} for {@code target}, or {@code null} for an empty index.
     *
     * <p>When an override iterator was supplied at construction, the scorer uses that iterator.
     * Otherwise, a fresh copy is used so the scorer's read cursor is independent from any outer
     * iteration. The scorer delegates to {@link RandomVectorScorer#score(int)} using the
     * iterator's current ordinal ({@code iterator.index()}).
     */
    @Override
    public VectorScorer scorer(byte[] target) throws IOException {
        if (size() == 0) return null;

        final FaissScorableByteVectorValues scorerCopy = copy();
        final RandomVectorScorer rvs = flatVectorsScorer.getRandomVectorScorer(similarityFunction, scorerCopy, target);
        final DocIndexIterator iterator = scorerCopy.iterator();

        return new VectorScorer() {
            @Override
            public float score() throws IOException {
                return rvs.score(iterator.index());
            }

            @Override
            public DocIdSetIterator iterator() {
                return iterator;
            }

            @Override
            public Bulk bulk(final DocIdSetIterator matchingDocs) {
                return Bulk.fromRandomScorerSparse(rvs, iterator, matchingDocs);
            }
        };
    }
}

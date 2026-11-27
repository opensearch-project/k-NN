/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

import java.io.IOException;

/**
 * Wraps the {@link FloatVectorValues} stored in a FAISS index and adds scoring capability via a {@link FlatVectorsScorer}.
 *
 * <p>All iteration/random-access methods delegate to the underlying values unchanged.
 * {@link #copy()} returns a fresh wrapper via {@code delegate.copy()} so that each scorer
 * holds an independent read cursor.
 *
 * <p>An optional {@link DocIndexIterator} can be supplied at construction time.
 * When present, both {@link #iterator()} and the scorer returned by {@link #scorer(float[])}
 * use this iterator instead of the delegate's default one.
 *
 * <p>{@link #scorer(float[])} hands the unwrapped delegate to the {@link FlatVectorsScorer}, so the scorer sees the
 * same values the memory-optimized searcher scores during ANN search (for example, memory-mapped FP16 or BF16 codes
 * scored with native SIMD).
 */
public class FaissScorableFloatVectorValues extends FloatVectorValues implements HasIndexSlice {
    private final FloatVectorValues delegate;
    private final FlatVectorsScorer flatVectorsScorer;
    private final VectorSimilarityFunction similarityFunction;
    private final DocIndexIterator overrideIterator;

    public FaissScorableFloatVectorValues(
        final FloatVectorValues delegate,
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
    public float[] vectorValue(int ord) throws IOException {
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
    public FaissScorableFloatVectorValues copy() throws IOException {
        return new FaissScorableFloatVectorValues(delegate.copy(), flatVectorsScorer, similarityFunction, overrideIterator);
    }

    @Override
    public int getVectorByteLength() {
        return delegate.getVectorByteLength();
    }

    @Override
    public VectorEncoding getEncoding() {
        return delegate.getEncoding();
    }

    @Override
    public VectorScorer rescorer(float[] target) throws IOException {
        return delegate.rescorer(target);
    }

    @Override
    public DocIndexIterator iterator() {
        if (overrideIterator != null) {
            return overrideIterator;
        }
        return delegate.iterator();
    }

    // ---- Scorer ----

    @Override
    public VectorScorer scorer(float[] target) throws IOException {
        if (size() == 0) return null;
        // Score the unwrapped FAISS values, so the flat vectors scorer picks the same scorer it uses for ANN search.
        final RandomVectorScorer rvs = flatVectorsScorer.getRandomVectorScorer(similarityFunction, delegate.copy(), target);
        final DocIndexIterator iterator = iterator();
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

    @Override
    public IndexInput getSlice() {
        if (delegate instanceof HasIndexSlice supportsIndexSlice) {
            return supportsIndexSlice.getSlice();
        }
        return null;
    }
}

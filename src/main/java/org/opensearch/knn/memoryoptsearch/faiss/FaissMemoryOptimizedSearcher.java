/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import com.google.common.annotations.VisibleForTesting;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.KnnVectorValues.DocIndexIterator;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.OrdinalTranslatedKnnCollector;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.common.RobustUniqueRandomIterator;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;
import org.opensearch.knn.memoryoptsearch.faiss.cagra.FaissCagraHNSW;

import java.io.IOException;

/**
 * This searcher directly reads FAISS index file via the provided {@link IndexInput} then perform vector search on it.
 */
public class FaissMemoryOptimizedSearcher implements VectorSearcher {

    /**
     * Exception thrown during warmup initialization when the search cannot proceed.
     * This encapsulates expected exceptions like NullPointerException (from null target vectors)
     * and UnsupportedOperationException (from vector encoding mismatches with quantized indices).
     */
    public static class WarmupInitializationException extends RuntimeException {
        public WarmupInitializationException(String message) {
            super(message);
        }
    }

    private final IndexInput indexInput;
    private final FaissIndex faissIndex;
    private final FlatVectorsScorer flatVectorsScorer;
    private final FaissHNSW hnsw;
    private final VectorSimilarityFunction vectorSimilarityFunction;
    private boolean isAdc;

    public FaissMemoryOptimizedSearcher(
        final IndexInput indexInput,
        final FaissIndex faissIndex,
        final FieldInfo fieldInfo,
        final FlatVectorsScorer flatVectorsScorer
    ) {
        this.indexInput = indexInput;
        this.faissIndex = faissIndex;
        final KNNVectorSimilarityFunction knnVectorSimilarityFunction = faissIndex.getVectorSimilarityFunction();

        if (knnVectorSimilarityFunction != KNNVectorSimilarityFunction.HAMMING) {
            vectorSimilarityFunction = knnVectorSimilarityFunction.getVectorSimilarityFunction();
        } else {
            vectorSimilarityFunction = null;
        }

        this.isAdc = FieldInfoExtractor.isAdc(fieldInfo);
        this.flatVectorsScorer = flatVectorsScorer;
        this.hnsw = extractFaissHnsw(faissIndex);
    }

    private static FaissHNSW extractFaissHnsw(final FaissIndex faissIndex) {
        if (faissIndex instanceof FaissIdMapIndex idMapIndex) {
            return idMapIndex.getFaissHnsw();
        }

        throw new IllegalArgumentException("Faiss index [" + faissIndex.getIndexType() + "] does not have HNSW as an index.");
    }

    @Override
    public void search(float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        final KnnVectorValues knnVectorValues = isAdc
            ? faissIndex.getByteValues(indexInput.clone())
            : faissIndex.getFloatValues(indexInput.clone());

        search(
            VectorEncoding.FLOAT32,
            flatVectorsScorer.getRandomVectorScorer(vectorSimilarityFunction, knnVectorValues, target),
            knnCollector,
            acceptDocs
        );
    }

    @Override
    public void search(byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        search(
            VectorEncoding.BYTE,
            flatVectorsScorer.getRandomVectorScorer(vectorSimilarityFunction, faissIndex.getByteValues(indexInput.clone()), target),
            knnCollector,
            acceptDocs
        );
    }

    /**
     * Returns a {@link FaissScorableByteVectorValues} that wraps the raw byte vectors from the
     * FAISS index with scoring support via {@link FlatVectorsScorer}.
     * <p>Each call creates a new instance backed by a fresh index input slice.
     */
    @Override
    public ByteVectorValues getByteVectorValues(DocIndexIterator iterator) throws IOException {
        return new FaissScorableByteVectorValues(
            faissIndex.getByteValues(indexInput.clone()),
            flatVectorsScorer,
            vectorSimilarityFunction,
            iterator
        );
    }

    @Override
    public void close() throws IOException {
        indexInput.close();
    }

    private void search(
        final VectorEncoding vectorEncoding,
        final RandomVectorScorer scorer,
        final KnnCollector knnCollector,
        final AcceptDocs acceptDocs
    ) throws IOException {
        if (faissIndex.getTotalNumberOfVectors() == 0 || knnCollector.k() == 0) {
            return;
        }

        if (!this.isAdc && faissIndex.getVectorEncoding() != vectorEncoding) {
            throw new IllegalArgumentException(
                "Search for vector encoding ["
                    + vectorEncoding
                    + "] is not supported in "
                    + "an index vector whose encoding is ["
                    + faissIndex.getVectorEncoding()
                    + "]"
            );
        }

        // Set up required components for vector search
        final KnnCollector collector = createKnnCollector(knnCollector, scorer);
        final Bits acceptedOrds = scorer.getAcceptOrds(acceptDocs.bits());

        if (knnCollector.k() < scorer.maxOrd()) {
            // Do ANN search with Lucene's HNSW graph searcher.
            HnswGraphSearcher.search(scorer, collector, new FaissHnswGraph(hnsw, indexInput.clone()), acceptedOrds);
        } else {
            // If k is larger than the number of vectors, we can just iterate over all vectors
            // and collect them.
            for (int i = 0; i < scorer.maxOrd(); i++) {
                if (acceptedOrds == null || acceptedOrds.get(i)) {
                    if (!knnCollector.earlyTerminated()) {
                        knnCollector.incVisitedCount(1);
                        knnCollector.collect(scorer.ordToDoc(i), scorer.score(i));
                    } else {
                        break;
                    }
                }
            }
        }
    }

    @VisibleForTesting
    KnnCollector createKnnCollector(final KnnCollector knnCollector, final RandomVectorScorer scorer) {
        final KnnCollector ordinalTranslatedKnnCollector = new OrdinalTranslatedKnnCollector(knnCollector, scorer::ordToDoc);

        if (hnsw instanceof FaissCagraHNSW cagraHNSW && (knnCollector.getSearchStrategy() instanceof KnnSearchStrategy.Seeded) == false) {
            // If there are provided entry points, then we should honor it and ensure searching to start based on them instead of
            // search with randomly selected points.
            return new KnnCollector.Decorator(ordinalTranslatedKnnCollector) {
                @Override
                public KnnSearchStrategy getSearchStrategy() {
                    return RandomEntryPointsKnnSearchStrategy.getInstance(
                        cagraHNSW.getNumBaseLevelSearchEntryPoints(),
                        cagraHNSW.getTotalNumberOfVectors(),
                        knnCollector.getSearchStrategy()
                    );
                }
            };
        }

        return ordinalTranslatedKnnCollector;
    }

    /**
     * Knn search strategy having a doc-id-iterator returning random document ids.
     * This is not designed for general purpose, it is particularly designed for populating random document ids for Cagra index.
     * Note that doc-id-iterator returns a random ids in `nextDoc` method without sorting, and might return duplicated ids.
     */
    static class RandomEntryPointsKnnSearchStrategy extends KnnSearchStrategy.Seeded {

        public static RandomEntryPointsKnnSearchStrategy getInstance(
            final int numberOfEntryPoints,
            final long totalNumberOfVectors,
            final KnnSearchStrategy originalStrategy
        ) {

            int entryPoints = getTotalNumberOfEntryPoints(numberOfEntryPoints, Math.toIntExact(totalNumberOfVectors));

            final DocIdSetIterator docIdSetIterator = generateRandomEntryPoints(entryPoints, Math.toIntExact(totalNumberOfVectors));

            return new RandomEntryPointsKnnSearchStrategy(docIdSetIterator, entryPoints, originalStrategy);
        }

        private RandomEntryPointsKnnSearchStrategy(
            final DocIdSetIterator entryPoints,
            final int numberOfEntryPoints,
            final KnnSearchStrategy originalStrategy
        ) {
            super(entryPoints, numberOfEntryPoints, originalStrategy);
        }

        private static int getTotalNumberOfEntryPoints(int numberOfEntryPoints, int totalVectors) {
            return numberOfEntryPoints >= totalVectors ? totalVectors : numberOfEntryPoints;
        }

        private static DocIdSetIterator generateRandomEntryPoints(final int numberOfEntryPoints, int totalNumberOfVectors) {
            if (numberOfEntryPoints >= totalNumberOfVectors) {
                return DocIdSetIterator.all(totalNumberOfVectors);
            }
            return new DocIdSetIterator() {
                final RobustUniqueRandomIterator robustUniqueRandomIterator = new RobustUniqueRandomIterator(
                    totalNumberOfVectors,
                    numberOfEntryPoints
                );

                @Override
                public int docID() {
                    throw new UnsupportedOperationException("DISI in RandomEntryPointsKnnSearchStrategy does not support docID()");
                }

                @Override
                public int nextDoc() {
                    return robustUniqueRandomIterator.next();
                }

                @Override
                public int advance(int targetDoc) {
                    throw new UnsupportedOperationException("DISI in RandomEntryPointsKnnSearchStrategy does not support advance(int)");
                }

                @Override
                public long cost() {
                    throw new UnsupportedOperationException("DISI in RandomEntryPointsKnnSearchStrategy does not support cost()");
                }
            };
        }
    }
}

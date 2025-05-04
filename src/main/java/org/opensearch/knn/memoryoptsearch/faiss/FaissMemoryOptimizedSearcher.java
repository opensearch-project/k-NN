/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOSupplier;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.OrdinalTranslatedKnnCollector;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;

import java.io.IOException;

/**
 * This searcher directly reads FAISS index file via the provided {@link IndexInput} then perform vector search on it.
 */
public class FaissMemoryOptimizedSearcher implements VectorSearcher {
    private final IndexInput indexInput;
    private final FaissIndex faissIndex;
    private final FlatVectorsScorer flatVectorsScorer;
    private final FaissHNSW hnsw;
    private final VectorSimilarityFunction vectorSimilarityFunction;

    public FaissMemoryOptimizedSearcher(IndexInput indexInput) throws IOException {
        this.indexInput = indexInput;
        this.faissIndex = FaissIndex.load(indexInput);
        final KNNVectorSimilarityFunction knnVectorSimilarityFunction = faissIndex.getVectorSimilarityFunction();
        this.flatVectorsScorer = FlatVectorsScorerProvider.getFlatVectorsScorer(knnVectorSimilarityFunction);
        if (knnVectorSimilarityFunction != KNNVectorSimilarityFunction.HAMMING) {
            vectorSimilarityFunction = knnVectorSimilarityFunction.getVectorSimilarityFunction();
        } else {
            vectorSimilarityFunction = null;
        }
        this.hnsw = extractFaissHnsw(faissIndex);
    }

    private static FaissHNSW extractFaissHnsw(final FaissIndex faissIndex) {
        if (faissIndex instanceof FaissIdMapIndex idMapIndex) {
            return idMapIndex.getNestedIndex().getHnsw();
        }

        throw new IllegalArgumentException("Faiss index [" + faissIndex.getIndexType() + "] does not have HNSW as an index.");
    }

    @Override
    public void search(float[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {
        search(
            VectorEncoding.FLOAT32,
            () -> flatVectorsScorer.getRandomVectorScorer(vectorSimilarityFunction, faissIndex.getFloatValues(indexInput), target),
            knnCollector,
            acceptDocs
        );
    }

    @Override
    public void search(byte[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {
        search(
            VectorEncoding.BYTE,
            () -> flatVectorsScorer.getRandomVectorScorer(vectorSimilarityFunction, faissIndex.getByteValues(indexInput), target),
            knnCollector,
            acceptDocs
        );
    }

    @Override
    public void close() throws IOException {
        indexInput.close();
    }

    private void search(
        final VectorEncoding vectorEncoding,
        final IOSupplier<RandomVectorScorer> scorerSupplier,
        final KnnCollector knnCollector,
        final Bits acceptDocs
    ) throws IOException {
        if (faissIndex.getTotalNumberOfVectors() == 0 || knnCollector.k() == 0) {
            return;
        }

        if (faissIndex.getVectorEncoding() != vectorEncoding) {
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
        final RandomVectorScorer scorer = scorerSupplier.get();
        final KnnCollector collector = new OrdinalTranslatedKnnCollector(knnCollector, scorer::ordToDoc);
        final Bits acceptedOrds = scorer.getAcceptOrds(acceptDocs);

        if (knnCollector.k() < scorer.maxOrd()) {
            // Do ANN search with Lucene's HNSW graph searcher.
            HnswGraphSearcher.search(scorer, collector, new FaissHnswGraph(hnsw, indexInput), acceptedOrds);
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
        }  // End if
    }
}

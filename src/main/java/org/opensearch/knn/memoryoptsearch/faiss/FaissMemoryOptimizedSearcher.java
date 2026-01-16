/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOSupplier;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.OrdinalTranslatedKnnCollector;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.jni.SimdVectorComputeService;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;
import org.opensearch.knn.memoryoptsearch.faiss.cagra.FaissCagraHNSW;

import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;

/**
 * This searcher directly reads FAISS index file via the provided {@link IndexInput} then perform vector search on it.
 */
public class FaissMemoryOptimizedSearcher implements VectorSearcher {

    private final IndexInput indexInput;
    private final FaissIndex faissIndex;
    private final FlatVectorsScorer flatVectorsScorer;
    private final FaissHNSW hnsw;
    private final VectorSimilarityFunction vectorSimilarityFunction;
    private final long fileSize;
    private boolean isAdc;
    private SimdVectorComputeService.SimilarityFunctionType nativeSimilarityFunctionType;

    public FaissMemoryOptimizedSearcher(final IndexInput indexInput, final FieldInfo fieldInfo) throws IOException {
        this.indexInput = indexInput;
        this.fileSize = indexInput.length();
        this.faissIndex = FaissIndex.load(indexInput);
        final KNNVectorSimilarityFunction knnVectorSimilarityFunction = faissIndex.getVectorSimilarityFunction();

        if (knnVectorSimilarityFunction != KNNVectorSimilarityFunction.HAMMING) {
            vectorSimilarityFunction = knnVectorSimilarityFunction.getVectorSimilarityFunction();
        } else {
            vectorSimilarityFunction = null;
        }

        this.isAdc = false;
        SpaceType spaceType = null;
        if (fieldInfo != null) {
            // Extract ADC info from fieldInfo to determine scorer.
            final QuantizationConfig quantizationConfig = FieldInfoExtractor.extractQuantizationConfig(fieldInfo);
            this.isAdc = quantizationConfig.isEnableADC();
            spaceType = isAdc ? SpaceType.getSpace(fieldInfo.getAttribute(KNNConstants.SPACE_TYPE)) : null;
        }

        this.flatVectorsScorer = FlatVectorsScorerProvider.getFlatVectorsScorer(knnVectorSimilarityFunction, isAdc, spaceType);

        this.hnsw = extractFaissHnsw(faissIndex);
        this.nativeSimilarityFunctionType = determineNativeFunctionType();
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
            ? faissIndex.getByteValues(getSlicedIndexInput())
            : faissIndex.getFloatValues(getSlicedIndexInput());
        final FloatVectorValues bottomKnnVectorValues = WrappedFloatVectorValues.getBottomFloatVectorValues(knnVectorValues);
        final boolean useNativeScoring = bottomKnnVectorValues instanceof MMapVectorValues;
        final IOSupplier<RandomVectorScorer> scorerSupplier;

        if (useNativeScoring) {
            // We can use native scoring.
            scorerSupplier = () -> new NativeRandomVectorScorer(
                target,
                knnVectorValues,
                (MMapVectorValues) bottomKnnVectorValues,
                nativeSimilarityFunctionType
            );
        } else {
            // Falling back to default scoring using pure Java.
            scorerSupplier = () -> flatVectorsScorer.getRandomVectorScorer(vectorSimilarityFunction, knnVectorValues, target);
        }

        search(VectorEncoding.FLOAT32, scorerSupplier, knnCollector, acceptDocs);
    }

    private SimdVectorComputeService.SimilarityFunctionType determineNativeFunctionType() {
        if (vectorSimilarityFunction == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT) {
            return SimdVectorComputeService.SimilarityFunctionType.FP16_MAXIMUM_INNER_PRODUCT;
        } else if (vectorSimilarityFunction == VectorSimilarityFunction.EUCLIDEAN) {
            return SimdVectorComputeService.SimilarityFunctionType.FP16_L2;
        }

        // At the moment, we only support FP16, it's fine to return null.
        return null;
    }

    @Override
    public void search(byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        search(
            VectorEncoding.BYTE,
            () -> flatVectorsScorer.getRandomVectorScorer(
                vectorSimilarityFunction,
                faissIndex.getByteValues(getSlicedIndexInput()),
                target
            ),
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
        final RandomVectorScorer scorer = scorerSupplier.get();
        final KnnCollector collector = createKnnCollector(knnCollector, scorer);
        final Bits acceptedOrds = scorer.getAcceptOrds(acceptDocs.bits());

        if (knnCollector.k() < scorer.maxOrd()) {
            // Do ANN search with Lucene's HNSW graph searcher.
            HnswGraphSearcher.search(scorer, collector, new FaissHnswGraph(hnsw, getSlicedIndexInput()), acceptedOrds);
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

    private IndexInput getSlicedIndexInput() throws IOException {
        return indexInput.slice("FaissMemoryOptimizedSearcher", 0, fileSize);
    }

    private KnnCollector createKnnCollector(final KnnCollector knnCollector, final RandomVectorScorer scorer) {
        final KnnCollector ordinalTranslatedKnnCollector = new OrdinalTranslatedKnnCollector(knnCollector, scorer::ordToDoc);

        if (hnsw instanceof FaissCagraHNSW cagraHNSW) {
            return new KnnCollector.Decorator(ordinalTranslatedKnnCollector) {
                @Override
                public KnnSearchStrategy getSearchStrategy() {
                    return new RandomEntryPointsKnnSearchStrategy(
                        cagraHNSW.getNumBaseLevelSearchEntryPoints(),
                        cagraHNSW.getTotalNumberOfVectors(),
                        knnCollector.getSearchStrategy()
                    );
                }
            };
        } else {
            return ordinalTranslatedKnnCollector;
        }
    }

    /**
     * Knn search strategy having a doc-id-iterator returning random document ids.
     * This is not designed for general purpose, it is particularly designed for populating random document ids for Cagra index.
     * Note that doc-id-iterator returns a random ids in `nextDoc` method without sorting, and might return duplicated ids.
     */
    static class RandomEntryPointsKnnSearchStrategy extends KnnSearchStrategy.Seeded {
        public RandomEntryPointsKnnSearchStrategy(
            final int numberOfEntryPoints,
            final long totalNumberOfVectors,
            final KnnSearchStrategy originalStrategy
        ) {
            super(
                generateRandomEntryPoints(numberOfEntryPoints, Math.toIntExact(totalNumberOfVectors)),
                numberOfEntryPoints,
                originalStrategy
            );
        }

        private static DocIdSetIterator generateRandomEntryPoints(final int numberOfEntryPoints, int totalNumberOfVectors) {
            return new DocIdSetIterator() {
                int numPopulatedVectors = 0;

                @Override
                public int docID() {
                    throw new UnsupportedOperationException("DISI in RandomEntryPointsKnnSearchStrategy does not support docID()");
                }

                @Override
                public int nextDoc() {
                    if (numPopulatedVectors < numberOfEntryPoints) {
                        ++numPopulatedVectors;
                        // It is fine to populate the same doc ids here, the same vectors will not be visited more than once with bitset.
                        return ThreadLocalRandom.current().nextInt(totalNumberOfVectors);
                    }

                    return NO_MORE_DOCS;
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

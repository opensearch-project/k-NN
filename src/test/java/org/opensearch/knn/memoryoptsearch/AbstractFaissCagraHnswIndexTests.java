/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.common.Randomness;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.IOConsumer;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.scorer.NativeEngines990KnnVectorsScorer;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissMemoryOptimizedSearcher;
import org.opensearch.knn.memoryoptsearch.faiss.FlatVectorsScorerProvider;

import java.nio.file.Files;
import java.nio.file.Path;
import org.opensearch.common.io.PathUtils;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.memoryoptsearch.FaissHNSWTests.loadHnswBinary;

public abstract class AbstractFaissCagraHnswIndexTests extends KNNTestCase {
    private static final int EF_SEARCH = 100;
    private static final int DIMENSION = 768;
    // Applying 32x quantization, one float will be quantized to 1 bit. As a result, one vector have 768 bits, which becomes 96 bytes.
    private static final int CODE_SIZE = 96;
    private static final int TOTAL_NUMBER_OF_VECTORS = 300;

    protected void doTestKNNSearch(
        final boolean isApproximateSearch,
        final VectorDataType vectorDataType,
        final KNNVectorSimilarityFunction similarityFunction
    ) {
        FieldInfo noADC = Mockito.mock(FieldInfo.class);
        Mockito.when(noADC.getAttribute(KNNConstants.QFRAMEWORK_CONFIG)).thenReturn(null);
        if (similarityFunction == KNNVectorSimilarityFunction.HAMMING) {
            Mockito.when(noADC.getAttribute(SPACE_TYPE)).thenReturn(SpaceType.HAMMING.getValue());
        } else if (similarityFunction == KNNVectorSimilarityFunction.EUCLIDEAN) {
            Mockito.when(noADC.getAttribute(SPACE_TYPE)).thenReturn(SpaceType.L2.getValue());
        } else if (similarityFunction == KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT) {
            Mockito.when(noADC.getAttribute(SPACE_TYPE)).thenReturn(SpaceType.INNER_PRODUCT.getValue());
        } else {
            throw new IllegalArgumentException("Unsupported similarity function: " + similarityFunction);
        }
        doTestWithIndexInput(input -> {
            // Instantiate memory optimized searcher
            final FaissIndex faissIndex = FaissIndex.load(input);
            final FaissMemoryOptimizedSearcher searcher = new FaissMemoryOptimizedSearcher(
                input,
                faissIndex,
                noADC,
                FlatVectorsScorerProvider.getFlatVectorsScorer(
                    noADC,
                    similarityFunction,
                    new NativeEngines990KnnVectorsScorer(FlatVectorScorerUtil.getLucene99FlatVectorsScorer())
                )
            );

            // Make collector
            final int k = isApproximateSearch ? EF_SEARCH : TOTAL_NUMBER_OF_VECTORS;
            final KnnCollector knnCollector = new TopKnnCollector(k, Integer.MAX_VALUE, KnnSearchStrategy.Hnsw.DEFAULT);

            // Build a query
            final Object query;
            if (vectorDataType == VectorDataType.FLOAT) {
                final float[] floatQuery = new float[DIMENSION];
                for (int i = 0; i < DIMENSION; i++) {
                    floatQuery[i] = Randomness.get().nextFloat();
                }
                query = floatQuery;
            } else if (vectorDataType == VectorDataType.BYTE) {
                final byte[] byteQuery = new byte[DIMENSION];
                for (int i = 0; i < DIMENSION; i++) {
                    byteQuery[i] = (byte) (Randomness.get().nextInt() & 0xFF);
                }
                query = byteQuery;
            } else {
                final byte[] binaryQuery = new byte[CODE_SIZE];
                for (int i = 0; i < CODE_SIZE; i++) {
                    binaryQuery[i] = (byte) (Randomness.get().nextInt() & 0xFF);
                }
                query = binaryQuery;
            }
            AcceptDocs acceptDocs = AcceptDocs.fromLiveDocs(null, TOTAL_NUMBER_OF_VECTORS);

            // Start searching
            if (vectorDataType == VectorDataType.FLOAT) {
                searcher.search((float[]) query, knnCollector, acceptDocs);
            } else {
                searcher.search((byte[]) query, knnCollector, acceptDocs);
            }
            final TopDocs topDocs = knnCollector.topDocs();
            final ScoreDoc[] scoreDocs = topDocs.scoreDocs;

            // Get answer
            input.seek(0);
            final FaissIndex answerIndex = FaissIndex.load(input);
            final Set<Integer> answerScoreDocs;
            if (vectorDataType == VectorDataType.FLOAT) {
                answerScoreDocs = calculateFloatAnswer((float[]) query, answerIndex.getFloatValues(input), k, similarityFunction);
            } else {
                answerScoreDocs = calculateByteOrBinaryAnswer((byte[]) query, answerIndex.getByteValues(input), k, similarityFunction);
            }

            // Validate search result
            int matchCount = 0;
            for (ScoreDoc sd : scoreDocs) {
                if (answerScoreDocs.contains(sd.doc)) {
                    ++matchCount;
                }
            }
            final float matchRatio = (float) matchCount / (float) k;
            assertTrue(matchRatio > 0.8);
        });
    }

    /**
     * Exact search scores the float vectors returned by {@link FaissMemoryOptimizedSearcher#getFloatVectorValues}. Checks
     * that they score every vector exactly as approximate search does on the same index, so exact search uses the same
     * scorer (for example native SIMD on memory-mapped FP16 codes) instead of a slower or different one.
     */
    protected void doTestExactSearchVectorsScoreLikeApproximateSearch(final KNNVectorSimilarityFunction similarityFunction) {
        final FieldInfo noADC = Mockito.mock(FieldInfo.class);
        Mockito.when(noADC.getAttribute(KNNConstants.QFRAMEWORK_CONFIG)).thenReturn(null);
        Mockito.when(noADC.getAttribute(SPACE_TYPE))
            .thenReturn(
                similarityFunction == KNNVectorSimilarityFunction.EUCLIDEAN ? SpaceType.L2.getValue() : SpaceType.INNER_PRODUCT.getValue()
            );
        doTestWithIndexInput(input -> {
            final FaissIndex faissIndex = FaissIndex.load(input);
            final FlatVectorsScorer flatVectorsScorer = FlatVectorsScorerProvider.getFlatVectorsScorer(
                noADC,
                similarityFunction,
                new NativeEngines990KnnVectorsScorer(FlatVectorScorerUtil.getLucene99FlatVectorsScorer())
            );
            final FaissMemoryOptimizedSearcher searcher = new FaissMemoryOptimizedSearcher(input, faissIndex, noADC, flatVectorsScorer);
            final float[] query = new float[DIMENSION];
            for (int i = 0; i < DIMENSION; i++) {
                query[i] = Randomness.get().nextFloat();
            }

            final FloatVectorValues exactSearchValues = searcher.getFloatVectorValues(denseIterator(TOTAL_NUMBER_OF_VECTORS));
            final VectorScorer exactSearchScorer = exactSearchValues.scorer(query);
            final RandomVectorScorer approximateSearchScorer = flatVectorsScorer.getRandomVectorScorer(
                similarityFunction.getVectorSimilarityFunction(),
                faissIndex.getFloatValues(input.clone()),
                query
            );

            final KnnVectorValues.DocIndexIterator iterator = (KnnVectorValues.DocIndexIterator) exactSearchScorer.iterator();
            int scored = 0;
            for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
                assertEquals(
                    "score of vector " + iterator.index(),
                    approximateSearchScorer.score(iterator.index()),
                    exactSearchScorer.score(),
                    0.0f
                );
                ++scored;
            }
            assertEquals(TOTAL_NUMBER_OF_VECTORS, scored);
        });
    }

    /**
     * The doc iterator exact search gets from the flat vectors in production: one document per vector, in ordinal order.
     */
    private static KnnVectorValues.DocIndexIterator denseIterator(final int size) {
        return new KnnVectorValues.DocIndexIterator() {
            private int doc = -1;

            @Override
            public int docID() {
                return doc;
            }

            @Override
            public int index() {
                return doc;
            }

            @Override
            public int nextDoc() {
                return advance(doc + 1);
            }

            @Override
            public int advance(final int target) {
                return doc = target >= size ? NO_MORE_DOCS : target;
            }

            @Override
            public long cost() {
                return size;
            }
        };
    }

    @SneakyThrows
    private Set<Integer> calculateByteOrBinaryAnswer(
        final byte[] query,
        final ByteVectorValues values,
        final int k,
        final KNNVectorSimilarityFunction similarityFunction
    ) {
        final List<ScoreDoc> scoreDocs = new ArrayList<>();
        for (int i = 0; i < TOTAL_NUMBER_OF_VECTORS; ++i) {
            final byte[] vector = values.vectorValue(i);
            scoreDocs.add(new ScoreDoc(i, similarityFunction.compare(query, vector)));
        }
        scoreDocs.sort((s1, s2) -> -Float.compare(s1.score, s2.score));
        return scoreDocs.subList(0, k).stream().mapToInt(s -> s.doc).boxed().collect(Collectors.toSet());
    }

    @SneakyThrows
    private Set<Integer> calculateFloatAnswer(
        final float[] query,
        final FloatVectorValues values,
        final int k,
        final KNNVectorSimilarityFunction similarityFunction
    ) {
        final List<ScoreDoc> scoreDocs = new ArrayList<>();
        for (int i = 0; i < TOTAL_NUMBER_OF_VECTORS; ++i) {
            final float[] vector = values.vectorValue(i);
            scoreDocs.add(new ScoreDoc(i, similarityFunction.compare(query, vector)));
        }
        scoreDocs.sort((s1, s2) -> -Float.compare(s1.score, s2.score));
        return scoreDocs.subList(0, k).stream().mapToInt(s -> s.doc).boxed().collect(Collectors.toSet());
    }

    protected void doTestLoadVectors(final VectorDataType vectorDataType, final Object firstVector, final Object lastVector) {
        doTestWithIndexInput(input -> {
            final FaissIndex faissIndex = FaissIndex.load(input);

            if (vectorDataType == VectorDataType.FLOAT) {
                final FloatVectorValues values = faissIndex.getFloatValues(input);

                // Validate the first vector
                float[] vector = values.vectorValue(0);
                assertArrayEquals((float[]) firstVector, vector, 1e-3f);

                // Validate the last vector
                vector = values.vectorValue(TOTAL_NUMBER_OF_VECTORS - 1);
                assertArrayEquals((float[]) lastVector, vector, 1e-3f);
            } else {
                final ByteVectorValues values = faissIndex.getByteValues(input);

                // Validate the first vector
                byte[] vector = values.vectorValue(0);
                assertArrayEquals((byte[]) firstVector, vector);

                // Validate the last vector
                vector = values.vectorValue(TOTAL_NUMBER_OF_VECTORS - 1);
                assertArrayEquals((byte[]) lastVector, vector);
            }
        });
    }

    @SneakyThrows
    private void doTestWithIndexInput(IOConsumer<IndexInput> indexInputConsumer) {
        // Non mmap input
        final IndexInput input = loadHnswBinary(getBinaryDataRelativePath());
        indexInputConsumer.accept(input);

        // Provide MMap input
        final Path tempDirPath = createTempDir();

        // Create a vector file in temp directory, we don't want to take a lock on common shared resource directory.
        final int tmpFileSize = (int) input.length();
        final Path tempFile = PathUtils.get(tempDirPath.toFile().getAbsolutePath(), "test.bin");
        final byte[] buffer = new byte[tmpFileSize];
        input.seek(0);
        input.readBytes(buffer, 0, tmpFileSize);
        Files.write(tempFile, buffer);

        // Create directory
        try (final Directory directory = new MMapDirectory(tempDirPath)) {
            try (final IndexInput indexInput = directory.openInput(tempFile.getFileName().toString(), IOContext.DEFAULT)) {
                indexInputConsumer.accept(indexInput);
            }
        }
    }

    protected abstract String getBinaryDataRelativePath();
}

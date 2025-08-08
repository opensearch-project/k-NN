/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOConsumer;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissMemoryOptimizedSearcher;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

import static org.mockito.Mockito.mock;
import static org.opensearch.knn.memoryoptsearch.FaissHNSWTests.loadHnswBinary;

public abstract class AbstractFaissCagraHnswIndexTests extends KNNTestCase {
    private static final FieldInfo NO_ADC_NEEDED = null;

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
        doTestWithIndexInput(input -> {
            FlatVectorsReaderWithFieldName flatVectorsReaderWithFieldName = mock(FlatVectorsReaderWithFieldName.class);

            // Instantiate memory optimized searcher
            final FaissMemoryOptimizedSearcher searcher = new FaissMemoryOptimizedSearcher(
                input,
                NO_ADC_NEEDED,
                flatVectorsReaderWithFieldName
            );

            // Make collector
            final int k = isApproximateSearch ? EF_SEARCH : TOTAL_NUMBER_OF_VECTORS;
            final KnnCollector knnCollector = new TopKnnCollector(k, Integer.MAX_VALUE);

            // Build a query
            final Object query;
            if (vectorDataType == VectorDataType.FLOAT) {
                final float[] floatQuery = new float[DIMENSION];
                for (int i = 0; i < DIMENSION; i++) {
                    floatQuery[i] = ThreadLocalRandom.current().nextFloat();
                }
                query = floatQuery;
            } else if (vectorDataType == VectorDataType.BYTE) {
                final byte[] byteQuery = new byte[DIMENSION];
                for (int i = 0; i < DIMENSION; i++) {
                    byteQuery[i] = (byte) (ThreadLocalRandom.current().nextInt() & 0xFF);
                }
                query = byteQuery;
            } else {
                final byte[] binaryQuery = new byte[CODE_SIZE];
                for (int i = 0; i < CODE_SIZE; i++) {
                    binaryQuery[i] = (byte) (ThreadLocalRandom.current().nextInt() & 0xFF);
                }
                query = binaryQuery;
            }

            // Start searching
            if (vectorDataType == VectorDataType.FLOAT) {
                searcher.search((float[]) query, knnCollector, null);
            } else {
                searcher.search((byte[]) query, knnCollector, null);
            }
            final TopDocs topDocs = knnCollector.topDocs();
            final ScoreDoc[] scoreDocs = topDocs.scoreDocs;

            // Get answer
            input.seek(0);
            final FaissIndex faissIndex = FaissIndex.load(input, flatVectorsReaderWithFieldName);
            final Set<Integer> answerScoreDocs;
            if (vectorDataType == VectorDataType.FLOAT) {
                answerScoreDocs = calculateFloatAnswer((float[]) query, faissIndex.getFloatValues(input), k, similarityFunction);
            } else {
                answerScoreDocs = calculateByteOrBinaryAnswer((byte[]) query, faissIndex.getByteValues(input), k, similarityFunction);
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
            FlatVectorsReaderWithFieldName flatVectorsReaderWithFieldName = mock(FlatVectorsReaderWithFieldName.class);
            final FaissIndex faissIndex = FaissIndex.load(input, flatVectorsReaderWithFieldName);

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
        final IndexInput input = loadHnswBinary(getBinaryDataRelativePath());
        indexInputConsumer.accept(input);
    }

    protected abstract String getBinaryDataRelativePath();
}

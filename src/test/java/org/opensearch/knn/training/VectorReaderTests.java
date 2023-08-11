/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.training;

import org.opensearch.core.action.ActionListener;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.KNNSingleNodeTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.stream.Collectors;

public class VectorReaderTests extends KNNSingleNodeTestCase {

    private final static int DEFAULT_LATCH_TIMEOUT = 100;
    private final static String DEFAULT_INDEX_NAME = "test-index";
    private final static String DEFAULT_FIELD_NAME = "test-field";
    private final static int DEFAULT_DIMENSION = 16;
    private final static int DEFAULT_NUM_VECTORS = 100;
    private final static int DEFAULT_MAX_VECTOR_COUNT = 10000;
    private final static int DEFAULT_SEARCH_SIZE = 10;

    public void testRead_valid_completeIndex() throws InterruptedException, ExecutionException, IOException {
        createIndex(DEFAULT_INDEX_NAME);
        createKnnIndexMapping(DEFAULT_INDEX_NAME, DEFAULT_FIELD_NAME, DEFAULT_DIMENSION);

        // Create list of random vectors and ingest
        Random random = new Random();
        List<Float[]> vectors = new ArrayList<>();
        for (int i = 0; i < DEFAULT_NUM_VECTORS; i++) {
            Float[] vector = random.doubles(DEFAULT_DIMENSION).boxed().map(Double::floatValue).toArray(Float[]::new);
            vectors.add(vector);
            addKnnDoc(DEFAULT_INDEX_NAME, Integer.toString(i), DEFAULT_FIELD_NAME, vector);
        }

        // Configure VectorReader
        ClusterService clusterService = node().injector().getInstance(ClusterService.class);
        VectorReader vectorReader = new VectorReader(client());

        // Read all vectors and confirm they match vectors
        TestVectorConsumer testVectorConsumer = new TestVectorConsumer();
        final CountDownLatch inProgressLatch = new CountDownLatch(1);
        vectorReader.read(
            clusterService,
            DEFAULT_INDEX_NAME,
            DEFAULT_FIELD_NAME,
            DEFAULT_MAX_VECTOR_COUNT,
            DEFAULT_SEARCH_SIZE,
            testVectorConsumer,
            createOnSearchResponseCountDownListener(inProgressLatch)
        );

        assertLatchDecremented(inProgressLatch);

        List<Float[]> consumedVectors = testVectorConsumer.getVectorsConsumed();
        assertEquals(DEFAULT_NUM_VECTORS, consumedVectors.size());

        List<Float> flatVectors = vectors.stream().flatMap(Arrays::stream).collect(Collectors.toList());
        List<Float> flatConsumedVectors = consumedVectors.stream().flatMap(Arrays::stream).collect(Collectors.toList());
        assertEquals(new HashSet<>(flatVectors), new HashSet<>(flatConsumedVectors));
    }

    public void testRead_valid_trainVectorsIngestedAsIntegers() throws IOException, ExecutionException, InterruptedException {
        createIndex(DEFAULT_INDEX_NAME);
        createKnnIndexMapping(DEFAULT_INDEX_NAME, DEFAULT_FIELD_NAME, DEFAULT_DIMENSION);

        // Create list of random vectors and ingest
        Random random = new Random();
        List<Integer[]> vectors = new ArrayList<>();
        for (int i = 0; i < DEFAULT_NUM_VECTORS; i++) {
            Integer[] vector = random.ints(DEFAULT_DIMENSION).boxed().toArray(Integer[]::new);
            vectors.add(vector);
            addKnnDoc(DEFAULT_INDEX_NAME, Integer.toString(i), DEFAULT_FIELD_NAME, vector);
        }

        // Configure VectorReader
        ClusterService clusterService = node().injector().getInstance(ClusterService.class);
        VectorReader vectorReader = new VectorReader(client());

        // Read all vectors and confirm they match vectors
        TestVectorConsumer testVectorConsumer = new TestVectorConsumer();
        final CountDownLatch inProgressLatch = new CountDownLatch(1);
        vectorReader.read(
            clusterService,
            DEFAULT_INDEX_NAME,
            DEFAULT_FIELD_NAME,
            DEFAULT_MAX_VECTOR_COUNT,
            DEFAULT_SEARCH_SIZE,
            testVectorConsumer,
            createOnSearchResponseCountDownListener(inProgressLatch)
        );

        assertLatchDecremented(inProgressLatch);

        List<Float[]> consumedVectors = testVectorConsumer.getVectorsConsumed();
        assertEquals(DEFAULT_NUM_VECTORS, consumedVectors.size());

        List<Float> flatVectors = vectors.stream().flatMap(Arrays::stream).map(Integer::floatValue).collect(Collectors.toList());
        List<Float> flatConsumedVectors = consumedVectors.stream().flatMap(Arrays::stream).collect(Collectors.toList());
        assertEquals(new HashSet<>(flatVectors), new HashSet<>(flatConsumedVectors));
    }

    public void testRead_valid_incompleteIndex() throws InterruptedException, ExecutionException, IOException {
        // Check if we get the right number of vectors if the index contains docs that are missing fields
        // Create an index with knn disabled
        createIndex(DEFAULT_INDEX_NAME);

        // Add a field mapping to the index
        createKnnIndexMapping(DEFAULT_INDEX_NAME, DEFAULT_FIELD_NAME, DEFAULT_DIMENSION);

        // Create list of random vectors and ingest
        Random random = new Random();
        List<Float[]> vectors = new ArrayList<>();
        for (int i = 0; i < DEFAULT_NUM_VECTORS; i++) {
            Float[] vector = random.doubles(DEFAULT_DIMENSION).boxed().map(Double::floatValue).toArray(Float[]::new);
            vectors.add(vector);
            addKnnDoc(DEFAULT_INDEX_NAME, Integer.toString(i), DEFAULT_FIELD_NAME, vector);
        }

        // Create documents that do not have fieldName for training
        int docsWithoutKNN = 100;
        String fieldNameWithoutKnn = "test-field-2";
        for (int i = 0; i < docsWithoutKNN; i++) {
            addDoc(DEFAULT_INDEX_NAME, Integer.toString(i + DEFAULT_NUM_VECTORS), fieldNameWithoutKnn, "dummyValue");
        }

        // Configure VectorReader
        ClusterService clusterService = node().injector().getInstance(ClusterService.class);
        VectorReader vectorReader = new VectorReader(client());

        // Read all vectors and confirm they match vectors
        TestVectorConsumer testVectorConsumer = new TestVectorConsumer();
        final CountDownLatch inProgressLatch = new CountDownLatch(1);
        vectorReader.read(
            clusterService,
            DEFAULT_INDEX_NAME,
            DEFAULT_FIELD_NAME,
            DEFAULT_MAX_VECTOR_COUNT,
            DEFAULT_SEARCH_SIZE,
            testVectorConsumer,
            createOnSearchResponseCountDownListener(inProgressLatch)
        );

        assertLatchDecremented(inProgressLatch);

        List<Float[]> consumedVectors = testVectorConsumer.getVectorsConsumed();
        assertEquals(DEFAULT_NUM_VECTORS, consumedVectors.size());

        List<Float> flatVectors = vectors.stream().flatMap(Arrays::stream).collect(Collectors.toList());
        List<Float> flatConsumedVectors = consumedVectors.stream().flatMap(Arrays::stream).collect(Collectors.toList());
        assertEquals(new HashSet<>(flatVectors), new HashSet<>(flatConsumedVectors));
    }

    public void testRead_valid_OnlyGetMaxVectors() throws InterruptedException, ExecutionException, IOException {
        // Check if we can limit the number of docs via max operation
        // Create an index with knn disabled
        int maxNumVectorsRead = 20;
        createIndex(DEFAULT_INDEX_NAME);

        // Add a field mapping to the index
        createKnnIndexMapping(DEFAULT_INDEX_NAME, DEFAULT_FIELD_NAME, DEFAULT_DIMENSION);

        // Create list of random vectors and ingest
        Random random = new Random();
        for (int i = 0; i < DEFAULT_NUM_VECTORS; i++) {
            Float[] vector = random.doubles(DEFAULT_DIMENSION).boxed().map(Double::floatValue).toArray(Float[]::new);
            addKnnDoc(DEFAULT_INDEX_NAME, Integer.toString(i), DEFAULT_FIELD_NAME, vector);
        }

        // Configure VectorReader
        ClusterService clusterService = node().injector().getInstance(ClusterService.class);
        VectorReader vectorReader = new VectorReader(client());

        // Read maxNumVectorsRead vectors
        TestVectorConsumer testVectorConsumer = new TestVectorConsumer();
        final CountDownLatch inProgressLatch = new CountDownLatch(1);
        vectorReader.read(
            clusterService,
            DEFAULT_INDEX_NAME,
            DEFAULT_FIELD_NAME,
            maxNumVectorsRead,
            DEFAULT_SEARCH_SIZE,
            testVectorConsumer,
            createOnSearchResponseCountDownListener(inProgressLatch)
        );

        assertLatchDecremented(inProgressLatch);

        List<Float[]> consumedVectors = testVectorConsumer.getVectorsConsumed();
        assertEquals(maxNumVectorsRead, consumedVectors.size());
    }

    public void testRead_invalid_maxVectorCount() {
        // Create the index
        createIndex(DEFAULT_INDEX_NAME);

        // Add a field mapping to the index
        createKnnIndexMapping(DEFAULT_INDEX_NAME, DEFAULT_FIELD_NAME, DEFAULT_DIMENSION);

        // Configure VectorReader
        ClusterService clusterService = node().injector().getInstance(ClusterService.class);
        VectorReader vectorReader = new VectorReader(client());

        int invalidMaxVectorCount = -10;
        expectThrows(
            ValidationException.class,
            () -> vectorReader.read(
                clusterService,
                DEFAULT_INDEX_NAME,
                DEFAULT_FIELD_NAME,
                invalidMaxVectorCount,
                DEFAULT_SEARCH_SIZE,
                null,
                null
            )
        );
    }

    public void testRead_invalid_searchSize() {
        // Create the index
        createIndex(DEFAULT_INDEX_NAME);

        // Add a field mapping to the index
        createKnnIndexMapping(DEFAULT_INDEX_NAME, DEFAULT_FIELD_NAME, DEFAULT_DIMENSION);

        // Configure VectorReader
        ClusterService clusterService = node().injector().getInstance(ClusterService.class);
        VectorReader vectorReader = new VectorReader(client());

        // Search size is negative
        int invalidSearchSize1 = -10;
        expectThrows(
            ValidationException.class,
            () -> vectorReader.read(
                clusterService,
                DEFAULT_INDEX_NAME,
                DEFAULT_FIELD_NAME,
                DEFAULT_MAX_VECTOR_COUNT,
                invalidSearchSize1,
                null,
                null
            )
        );

        // Search size is greater than 10000
        int invalidSearchSize2 = 20000;
        expectThrows(
            ValidationException.class,
            () -> vectorReader.read(
                clusterService,
                DEFAULT_INDEX_NAME,
                DEFAULT_FIELD_NAME,
                DEFAULT_MAX_VECTOR_COUNT,
                invalidSearchSize2,
                null,
                null
            )
        );
    }

    public void testRead_invalid_indexDoesNotExist() {
        // Check that read throws a validation exception when the index does not exist
        // Configure VectorReader
        ClusterService clusterService = node().injector().getInstance(ClusterService.class);
        VectorReader vectorReader = new VectorReader(client());

        // Should throw a validation exception because index does not exist
        expectThrows(
            ValidationException.class,
            () -> vectorReader.read(
                clusterService,
                DEFAULT_INDEX_NAME,
                DEFAULT_FIELD_NAME,
                DEFAULT_MAX_VECTOR_COUNT,
                DEFAULT_SEARCH_SIZE,
                null,
                null
            )
        );
    }

    public void testRead_invalid_fieldDoesNotExist() {
        // Check that read throws a validation exception when the field does not exist
        createIndex(DEFAULT_INDEX_NAME);

        // Configure VectorReader
        ClusterService clusterService = node().injector().getInstance(ClusterService.class);
        VectorReader vectorReader = new VectorReader(client());

        // Should throw a validation exception because field is not k-NN
        expectThrows(
            ValidationException.class,
            () -> vectorReader.read(
                clusterService,
                DEFAULT_INDEX_NAME,
                DEFAULT_FIELD_NAME,
                DEFAULT_MAX_VECTOR_COUNT,
                DEFAULT_SEARCH_SIZE,
                null,
                null
            )
        );
    }

    public void testRead_invalid_fieldIsNotKnn() throws InterruptedException, ExecutionException, IOException {
        // Check that read throws a validation exception when the field does not exist
        createIndex(DEFAULT_INDEX_NAME);
        addDoc(DEFAULT_INDEX_NAME, "test-id", DEFAULT_FIELD_NAME, "dummy");

        // Configure VectorReader
        ClusterService clusterService = node().injector().getInstance(ClusterService.class);
        VectorReader vectorReader = new VectorReader(client());

        // Should throw a validation exception because field does not exist
        expectThrows(
            ValidationException.class,
            () -> vectorReader.read(
                clusterService,
                DEFAULT_INDEX_NAME,
                DEFAULT_FIELD_NAME,
                DEFAULT_MAX_VECTOR_COUNT,
                DEFAULT_SEARCH_SIZE,
                null,
                null
            )
        );
    }

    private static class TestVectorConsumer implements Consumer<List<Float[]>> {

        List<Float[]> vectorsConsumed;

        TestVectorConsumer() {
            vectorsConsumed = new ArrayList<>();
        }

        @Override
        public void accept(List<Float[]> vectors) {
            vectorsConsumed.addAll(vectors);
        }

        public List<Float[]> getVectorsConsumed() {
            return vectorsConsumed;
        }
    }

    private void assertLatchDecremented(CountDownLatch countDownLatch) throws InterruptedException {
        assertTrue(countDownLatch.await(DEFAULT_LATCH_TIMEOUT, TimeUnit.SECONDS));
    }

    private ActionListener<SearchResponse> createOnSearchResponseCountDownListener(CountDownLatch countDownLatch) {
        return ActionListener.wrap(response -> countDownLatch.countDown(), Throwable::printStackTrace);
    }
}

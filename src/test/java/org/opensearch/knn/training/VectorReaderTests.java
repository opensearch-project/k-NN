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

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.action.ActionListener;
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

    public static Logger logger = LogManager.getLogger(VectorReaderTests.class);

    public void testRead_valid_completeIndex() throws InterruptedException, ExecutionException, IOException {
        // Create an index with knn disabled
        String indexName = "test-index";
        String fieldName = "test-field";
        int dim = 16;
        int numVectors = 100;
        createIndex(indexName);

        // Add a field mapping to the index
        createKnnIndexMapping(indexName, fieldName, dim);

        // Create list of random vectors and ingest
        Random random = new Random();
        List<Float[]> vectors = new ArrayList<>();
        for (int i = 0; i < numVectors; i++) {
            Float[] vector = new Float[dim];

            for (int j = 0; j < dim; j++) {
                vector[j] = random.nextFloat();
            }

            vectors.add(vector);

            addKnnDoc(indexName, Integer.toString(i), fieldName, vector);
        }

        // Configure VectorReader
        ClusterService clusterService = node().injector().getInstance(ClusterService.class);
        VectorReader vectorReader = new VectorReader(client());

        // Read all vectors and confirm they match vectors
        TestVectorConsumer testVectorConsumer = new TestVectorConsumer();
        final CountDownLatch inProgressLatch1 = new CountDownLatch(1);
        vectorReader.read(clusterService, indexName, fieldName, 10000, 10, testVectorConsumer,
                ActionListener.wrap(response -> inProgressLatch1.countDown(), e -> fail(e.toString())));

        assertTrue(inProgressLatch1.await(100, TimeUnit.SECONDS));

        List<Float[]> consumedVectors = testVectorConsumer.getVectorsConsumed();
        assertEquals(numVectors, consumedVectors.size());

        List<Float> flatVectors = vectors.stream().flatMap(Arrays::stream).collect(Collectors.toList());
        List<Float> flatConsumedVectors = consumedVectors.stream().flatMap(Arrays::stream).collect(Collectors.toList());
        assertEquals(new HashSet<>(flatVectors), new HashSet<>(flatConsumedVectors));
    }

    public void testRead_valid_incompleteIndex() throws InterruptedException, ExecutionException, IOException {
        // Check if we get the right number of vectors if the index contains docs that are missing fields
        // Create an index with knn disabled
        String indexName = "test-index";
        String fieldName = "test-field";
        int dim = 16;
        int numVectors = 100;
        createIndex(indexName);

        // Add a field mapping to the index
        createKnnIndexMapping(indexName, fieldName, dim);

        // Create list of random vectors and ingest
        Random random = new Random();
        List<Float[]> vectors = new ArrayList<>();
        for (int i = 0; i < numVectors; i++) {
            Float[] vector = new Float[dim];

            for (int j = 0; j < dim; j++) {
                vector[j] = random.nextFloat();
            }

            vectors.add(vector);

            addKnnDoc(indexName, Integer.toString(i ), fieldName, vector);
        }

        // Create documents that do not have fieldName for training
        int docsWithoutKNN = 100;
        String fieldNameWithoutKnn = "test-field-2";
        for (int i = 0; i < docsWithoutKNN; i++) {
            addDoc(indexName, Integer.toString(i + numVectors), fieldNameWithoutKnn, "dummyValue");
        }

        // Configure VectorReader
        ClusterService clusterService = node().injector().getInstance(ClusterService.class);
        VectorReader vectorReader = new VectorReader(client());

        // Read all vectors and confirm they match vectors
        TestVectorConsumer testVectorConsumer = new TestVectorConsumer();
        final CountDownLatch inProgressLatch1 = new CountDownLatch(1);
        vectorReader.read(clusterService, indexName, fieldName, 10000, 10, testVectorConsumer,
                ActionListener.wrap(response -> inProgressLatch1.countDown(), e -> fail(e.toString())));

        assertTrue(inProgressLatch1.await(100, TimeUnit.SECONDS));

        List<Float[]> consumedVectors = testVectorConsumer.getVectorsConsumed();
        assertEquals(numVectors, consumedVectors.size());

        List<Float> flatVectors = vectors.stream().flatMap(Arrays::stream).collect(Collectors.toList());
        List<Float> flatConsumedVectors = consumedVectors.stream().flatMap(Arrays::stream).collect(Collectors.toList());
        assertEquals(new HashSet<>(flatVectors), new HashSet<>(flatConsumedVectors));
    }

    public void testRead_valid_OnlyGetMaxVectors() throws InterruptedException, ExecutionException, IOException {
        // Check if we can limit the number of docs via max operation
        // Create an index with knn disabled
        String indexName = "test-index";
        String fieldName = "test-field";
        int dim = 16;
        int numVectorsIndex = 100;
        int maxNumVectorsRead = 20;
        createIndex(indexName);

        // Add a field mapping to the index
        createKnnIndexMapping(indexName, fieldName, dim);

        // Create list of random vectors and ingest
        Random random = new Random();
        for (int i = 0; i < numVectorsIndex; i++) {
            Float[] vector = new Float[dim];

            for (int j = 0; j < dim; j++) {
                vector[j] = random.nextFloat();
            }

            addKnnDoc(indexName, Integer.toString(i ), fieldName, vector);
        }

        // Configure VectorReader
        ClusterService clusterService = node().injector().getInstance(ClusterService.class);
        VectorReader vectorReader = new VectorReader(client());

        // Read maxNumVectorsRead vectors
        TestVectorConsumer testVectorConsumer = new TestVectorConsumer();
        final CountDownLatch inProgressLatch1 = new CountDownLatch(1);
        vectorReader.read(clusterService, indexName, fieldName, maxNumVectorsRead, 10, testVectorConsumer,
                ActionListener.wrap(response -> inProgressLatch1.countDown(), e -> fail(e.toString())));

        assertTrue(inProgressLatch1.await(100, TimeUnit.SECONDS));

        List<Float[]> consumedVectors = testVectorConsumer.getVectorsConsumed();
        assertEquals(maxNumVectorsRead, consumedVectors.size());
    }

    public void testRead_invalid_maxVectorCount() {
        // Create the index
        String indexName = "test-index";
        String fieldName = "test-field";
        int dim = 16;
        createIndex(indexName);

        // Add a field mapping to the index
        createKnnIndexMapping(indexName, fieldName, dim);

        // Configure VectorReader
        ClusterService clusterService = node().injector().getInstance(ClusterService.class);
        VectorReader vectorReader = new VectorReader(client());

        expectThrows(ValidationException.class, () -> vectorReader.read(clusterService, indexName, fieldName, -10, 10, null, null));
    }

    public void testRead_invalid_searchSize() {
        // Create the index
        String indexName = "test-index";
        String fieldName = "test-field";
        int dim = 16;
        createIndex(indexName);

        // Add a field mapping to the index
        createKnnIndexMapping(indexName, fieldName, dim);

        // Configure VectorReader
        ClusterService clusterService = node().injector().getInstance(ClusterService.class);
        VectorReader vectorReader = new VectorReader(client());

        // Search size is negative
        expectThrows(ValidationException.class, () -> vectorReader.read(clusterService, indexName, fieldName, 100, -10, null, null));

        // Search size is greater than 10000
        expectThrows(ValidationException.class, () -> vectorReader.read(clusterService, indexName, fieldName, 100, 20000, null, null));
    }

    public void testRead_invalid_indexDoesNotExist() {
        // Check that read throws a validation exception when the index does not exist
        String indexName = "test-index";
        String fieldName = "test-field";

        // Configure VectorReader
        ClusterService clusterService = node().injector().getInstance(ClusterService.class);
        VectorReader vectorReader = new VectorReader(client());

        // Should throw a validation exception because index does not exist
        expectThrows(ValidationException.class, () -> vectorReader.read(clusterService, indexName, fieldName, 10000, 10, null, null));
    }

    public void testRead_invalid_fieldDoesNotExist() {
        // Check that read throws a validation exception when the field does not exist
        String indexName = "test-index";
        String fieldName = "test-field";
        createIndex(indexName);

        // Configure VectorReader
        ClusterService clusterService = node().injector().getInstance(ClusterService.class);
        VectorReader vectorReader = new VectorReader(client());

        // Should throw a validation exception because field is not k-NN
        expectThrows(ValidationException.class, () -> vectorReader.read(clusterService, indexName, fieldName, 10000, 10, null, null));
    }

    public void testRead_invalid_fieldIsNotKnn() throws InterruptedException, ExecutionException, IOException {
        // Check that read throws a validation exception when the field does not exist
        String indexName = "test-index";
        String fieldName = "test-field";
        createIndex(indexName);
        addDoc(indexName, "test-id", fieldName, "dummy");

        // Configure VectorReader
        ClusterService clusterService = node().injector().getInstance(ClusterService.class);
        VectorReader vectorReader = new VectorReader(client());

        // Should throw a validation exception because field does not exist
        expectThrows(ValidationException.class, () -> vectorReader.read(clusterService, indexName, fieldName, 10000, 10, null, null));
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
}
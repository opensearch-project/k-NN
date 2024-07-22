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

package org.opensearch.knn.index.memory;

import com.google.common.collect.ImmutableMap;
import org.opensearch.core.action.ActionListener;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.jni.JNICommons;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.training.FloatTrainingDataConsumer;
import org.opensearch.knn.training.VectorReader;
import org.opensearch.watcher.ResourceWatcherService;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;

import static org.mockito.Mockito.any;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;

public class NativeMemoryLoadStrategyTests extends KNNTestCase {

    public void testIndexLoadStrategy_load() throws IOException {
        // Create basic nmslib HNSW index
        Path dir = createTempDir();
        KNNEngine knnEngine = KNNEngine.NMSLIB;
        String indexName = "test1" + knnEngine.getExtension();
        String path = dir.resolve(indexName).toAbsolutePath().toString();
        int numVectors = 10;
        int dimension = 10;
        int[] ids = new int[numVectors];
        float[][] vectors = new float[numVectors][dimension];
        for (int i = 0; i < numVectors; i++) {
            ids[i] = i;
            Arrays.fill(vectors[i], 1f);
        }
        Map<String, Object> parameters = ImmutableMap.of(KNNConstants.SPACE_TYPE, SpaceType.DEFAULT.getValue());
        long memoryAddress = JNICommons.storeVectorData(0, vectors, numVectors * dimension);
        JNIService.createIndex(ids, memoryAddress, dimension, path, parameters, knnEngine);

        // Setup mock resource manager
        ResourceWatcherService resourceWatcherService = mock(ResourceWatcherService.class);
        doReturn(null).when(resourceWatcherService).add(any());
        NativeMemoryLoadStrategy.IndexLoadStrategy.initialize(resourceWatcherService);

        NativeMemoryEntryContext.IndexEntryContext indexEntryContext = new NativeMemoryEntryContext.IndexEntryContext(
            path,
            NativeMemoryLoadStrategy.IndexLoadStrategy.getInstance(),
            parameters,
            "test"
        );

        // Load
        NativeMemoryAllocation.IndexAllocation indexAllocation = NativeMemoryLoadStrategy.IndexLoadStrategy.getInstance()
            .load(indexEntryContext);

        // Confirm that the file was loaded by querying
        float[] query = new float[dimension];
        Arrays.fill(query, numVectors + 1);
        KNNQueryResult[] results = JNIService.queryIndex(indexAllocation.getMemoryAddress(), query, 2, null, knnEngine, null, 0, null);
        assertTrue(results.length > 0);
    }

    public void testLoad_whenFaissBinary_thenSuccess() throws IOException {
        Path dir = createTempDir();
        KNNEngine knnEngine = KNNEngine.FAISS;
        String indexName = "test1" + knnEngine.getExtension();
        String path = dir.resolve(indexName).toAbsolutePath().toString();
        int numVectors = 10;
        int dimension = 8;
        int dataLength = dimension / 8;
        int[] ids = new int[numVectors];
        byte[][] vectors = new byte[numVectors][dataLength];
        for (int i = 0; i < numVectors; i++) {
            ids[i] = i;
            vectors[i][0] = 1;
        }
        Map<String, Object> parameters = ImmutableMap.of(
            KNNConstants.SPACE_TYPE,
            SpaceType.HAMMING.getValue(),
            KNNConstants.INDEX_DESCRIPTION_PARAMETER,
            "BHNSW32",
            KNNConstants.VECTOR_DATA_TYPE_FIELD,
            VectorDataType.BINARY.getValue()
        );
        long memoryAddress = JNICommons.storeByteVectorData(0, vectors, numVectors);
        JNIService.createIndex(ids, memoryAddress, dimension, path, parameters, knnEngine);

        // Setup mock resource manager
        ResourceWatcherService resourceWatcherService = mock(ResourceWatcherService.class);
        doReturn(null).when(resourceWatcherService).add(any());
        NativeMemoryLoadStrategy.IndexLoadStrategy.initialize(resourceWatcherService);

        NativeMemoryEntryContext.IndexEntryContext indexEntryContext = new NativeMemoryEntryContext.IndexEntryContext(
            path,
            NativeMemoryLoadStrategy.IndexLoadStrategy.getInstance(),
            parameters,
            "test"
        );

        // Load
        NativeMemoryAllocation.IndexAllocation indexAllocation = NativeMemoryLoadStrategy.IndexLoadStrategy.getInstance()
            .load(indexEntryContext);

        // Verify
        assertTrue(indexAllocation.isBinaryIndex());

        // Confirm that the file was loaded by querying
        byte[] query = { 1 };
        KNNQueryResult[] results = JNIService.queryBinaryIndex(
            indexAllocation.getMemoryAddress(),
            query,
            2,
            null,
            knnEngine,
            null,
            0,
            null
        );
        assertTrue(results.length > 0);
    }

    @SuppressWarnings("unchecked")
    public void testTrainingLoadStrategy_load() {
        // Mock the vector reader so that on read, it waits 2 seconds, transfers vectors to the consumer, and then calls
        // listener onResponse to release the write lock
        VectorReader vectorReader = mock(VectorReader.class);
        ArrayList<Float[]> vectors = new ArrayList<>();
        vectors.add(new Float[] { 1.0F, 2.0F });
        logger.info("J0");
        doAnswer(invocationOnMock -> {
            logger.info("J1");
            FloatTrainingDataConsumer floatTrainingDataConsumer = (FloatTrainingDataConsumer) invocationOnMock.getArguments()[5];
            ActionListener<SearchResponse> listener = (ActionListener<SearchResponse>) invocationOnMock.getArguments()[6];
            Thread thread = new Thread(() -> {
                try {
                    Thread.sleep(2000);
                    floatTrainingDataConsumer.accept(vectors); // Transfer some floats
                    listener.onResponse(null);
                } catch (InterruptedException e) {
                    listener.onFailure(null);
                    fail("Failed due to interuption");
                }
            });

            thread.start();
            return null;
        }).when(vectorReader).read(eq(null), eq("test"), eq("test"), eq(0), eq(0), any(), any());

        NativeMemoryLoadStrategy.TrainingLoadStrategy.initialize(vectorReader);

        NativeMemoryEntryContext.TrainingDataEntryContext trainingDataEntryContext = new NativeMemoryEntryContext.TrainingDataEntryContext(
            0,
            "test",
            "test",
            NativeMemoryLoadStrategy.TrainingLoadStrategy.getInstance(),
            null,
            0,
            0,
            VectorDataType.FLOAT
        );

        // Load the allocation. Initially, the memory address should be 0. However, after the readlock is obtained,
        // the memory address should not be 0.
        NativeMemoryAllocation.TrainingDataAllocation trainingDataAllocation = NativeMemoryLoadStrategy.TrainingLoadStrategy.getInstance()
            .load(trainingDataEntryContext);
        assertEquals(0, trainingDataAllocation.getMemoryAddress());
        trainingDataAllocation.readLock();
        assertNotEquals(0, trainingDataAllocation.getMemoryAddress());
        trainingDataAllocation.readUnlock();
    }
}

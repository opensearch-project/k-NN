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
import lombok.SneakyThrows;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.IndexUtil;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.jni.JNICommons;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.watcher.FileWatcher;
import org.opensearch.watcher.WatcherHandle;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.mock;

public class NativeMemoryAllocationTests extends KNNTestCase {

    private int testLockValue1;
    private int testLockValue2;
    private int testLockValue3;
    private int testLockValue4;

    public void testIndexAllocation_close() throws InterruptedException {
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
        long vectorMemoryAddress = JNICommons.storeVectorData(0, vectors, numVectors * dimension);
        JNIService.createIndex(ids, vectorMemoryAddress, dimension, path, parameters, knnEngine);

        // Load index into memory
        long memoryAddress = JNIService.loadIndex(path, parameters, knnEngine);

        @SuppressWarnings("unchecked")
        WatcherHandle<FileWatcher> watcherHandle = (WatcherHandle<FileWatcher>) mock(WatcherHandle.class);
        doNothing().when(watcherHandle).stop();

        ExecutorService executorService = Executors.newSingleThreadExecutor();
        NativeMemoryAllocation.IndexAllocation indexAllocation = new NativeMemoryAllocation.IndexAllocation(
            executorService,
            memoryAddress,
            IndexUtil.getFileSizeInKB(path),
            knnEngine,
            path,
            "test",
            watcherHandle
        );

        indexAllocation.close();

        Thread.sleep(1000 * 2);
        indexAllocation.writeLock();
        assertTrue(indexAllocation.isClosed());
        indexAllocation.writeUnlock();

        indexAllocation.close();

        Thread.sleep(1000 * 2);
        indexAllocation.writeLock();
        assertTrue(indexAllocation.isClosed());
        indexAllocation.writeUnlock();

        executorService.shutdown();
    }

    @SneakyThrows
    public void testClose_whenBinaryFiass_thenSuccess() {
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
        long vectorMemoryAddress = JNICommons.storeByteVectorData(0, vectors, numVectors * dataLength);
        JNIService.createIndex(ids, vectorMemoryAddress, dimension, path, parameters, knnEngine);

        // Load index into memory
        long memoryAddress = JNIService.loadIndex(path, parameters, knnEngine);

        @SuppressWarnings("unchecked")
        WatcherHandle<FileWatcher> watcherHandle = (WatcherHandle<FileWatcher>) mock(WatcherHandle.class);
        doNothing().when(watcherHandle).stop();

        ExecutorService executorService = Executors.newSingleThreadExecutor();
        NativeMemoryAllocation.IndexAllocation indexAllocation = new NativeMemoryAllocation.IndexAllocation(
            executorService,
            memoryAddress,
            IndexUtil.getFileSizeInKB(path),
            knnEngine,
            path,
            "test",
            watcherHandle,
            null,
            true
        );

        indexAllocation.close();

        Thread.sleep(1000 * 2);
        indexAllocation.writeLock();
        assertTrue(indexAllocation.isClosed());
        indexAllocation.writeUnlock();

        indexAllocation.close();

        Thread.sleep(1000 * 2);
        indexAllocation.writeLock();
        assertTrue(indexAllocation.isClosed());
        indexAllocation.writeUnlock();

        executorService.shutdown();
    }

    public void testIndexAllocation_getMemoryAddress() {
        long memoryAddress = 12;
        NativeMemoryAllocation.IndexAllocation indexAllocation = new NativeMemoryAllocation.IndexAllocation(
            null,
            memoryAddress,
            0,
            null,
            "test",
            "test",
            null
        );

        assertEquals(memoryAddress, indexAllocation.getMemoryAddress());
    }

    public void testIndexAllocation_readLock() throws InterruptedException {
        // To test the readLock, we grab the readLock in the main thread and then start a thread that grabs the write
        // lock and updates testLockValue1. We ensure that the value is not updated until after we release the readLock
        NativeMemoryAllocation.IndexAllocation indexAllocation = new NativeMemoryAllocation.IndexAllocation(
            null,
            0,
            0,
            null,
            "test",
            "test",
            null
        );

        int initialValue = 10;
        int finalValue = 16;
        testLockValue1 = initialValue;

        indexAllocation.readLock();

        Thread thread = new Thread(() -> {
            indexAllocation.writeLock();
            testLockValue1 = finalValue;
            indexAllocation.writeUnlock();
        });

        thread.start();
        Thread.sleep(1000);

        assertEquals(initialValue, testLockValue1);
        indexAllocation.readUnlock();

        Thread.sleep(1000);
        assertEquals(finalValue, testLockValue1);
    }

    public void testIndexAllocation_writeLock() throws InterruptedException {
        // To test the writeLock, we first grab the writeLock in the main thread. Then we start another thread that
        // grabs the readLock and asserts testLockValue2 has been updated. Next in the main thread, we update the value
        // and release the writeLock.
        NativeMemoryAllocation.IndexAllocation indexAllocation = new NativeMemoryAllocation.IndexAllocation(
            null,
            0,
            0,
            null,
            "test",
            "test",
            null
        );

        int initialValue = 10;
        int finalValue = 16;
        testLockValue2 = initialValue;

        indexAllocation.writeLock();

        Thread thread = new Thread(() -> {
            indexAllocation.readLock();
            assertEquals(finalValue, testLockValue2);
            indexAllocation.readUnlock();
        });

        thread.start();
        Thread.sleep(1000);

        testLockValue2 = finalValue;
        indexAllocation.writeUnlock();

        Thread.sleep(1000);
    }

    public void testIndexAllocation_getSize() {
        int size = 12;
        NativeMemoryAllocation.IndexAllocation indexAllocation = new NativeMemoryAllocation.IndexAllocation(
            null,
            0,
            size,
            null,
            "test",
            "test",
            null
        );

        assertEquals(size, indexAllocation.getSizeInKB());
    }

    public void testIndexAllocation_getKnnEngine() {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        NativeMemoryAllocation.IndexAllocation indexAllocation = new NativeMemoryAllocation.IndexAllocation(
            null,
            0,
            0,
            knnEngine,
            "test",
            "test",
            null
        );

        assertEquals(knnEngine, indexAllocation.getKnnEngine());
    }

    public void testIndexAllocation_getIndexPath() {
        String indexPath = "test-path";
        NativeMemoryAllocation.IndexAllocation indexAllocation = new NativeMemoryAllocation.IndexAllocation(
            null,
            0,
            0,
            null,
            indexPath,
            "test",
            null
        );

        assertEquals(indexPath, indexAllocation.getIndexPath());
    }

    public void testIndexAllocation_getOsIndexName() {
        String osIndexName = "test-index";
        NativeMemoryAllocation.IndexAllocation indexAllocation = new NativeMemoryAllocation.IndexAllocation(
            null,
            0,
            0,
            null,
            "test",
            osIndexName,
            null
        );

        assertEquals(osIndexName, indexAllocation.getOpenSearchIndexName());
    }

    public void testTrainingDataAllocation_close() throws InterruptedException {
        // Create basic nmslib HNSW index
        int numVectors = 10;
        int dimension = 10;
        float[][] vectors = new float[numVectors][dimension];
        for (int i = 0; i < numVectors; i++) {
            Arrays.fill(vectors[i], 1f);
        }
        long memoryAddress = JNIService.transferVectors(0, vectors);

        ExecutorService executorService = Executors.newSingleThreadExecutor();
        NativeMemoryAllocation.TrainingDataAllocation trainingDataAllocation = new NativeMemoryAllocation.TrainingDataAllocation(
            executorService,
            memoryAddress,
            0,
            VectorDataType.FLOAT
        );

        trainingDataAllocation.close();

        Thread.sleep(1000 * 2);
        trainingDataAllocation.writeLock();
        assertTrue(trainingDataAllocation.isClosed());
        trainingDataAllocation.writeUnlock();

        trainingDataAllocation.close();

        Thread.sleep(1000 * 2);
        trainingDataAllocation.writeLock();
        assertTrue(trainingDataAllocation.isClosed());
        trainingDataAllocation.writeUnlock();

        executorService.shutdown();
    }

    public void testTrainingDataAllocation_getMemoryAddress() {
        long memoryAddress = 12;

        NativeMemoryAllocation.TrainingDataAllocation trainingDataAllocation = new NativeMemoryAllocation.TrainingDataAllocation(
            null,
            memoryAddress,
            0,
            VectorDataType.FLOAT
        );

        assertEquals(memoryAddress, trainingDataAllocation.getMemoryAddress());
    }

    public void testTrainingDataAllocation_readLock() throws InterruptedException {
        // To test readLock functionality, we first lock reads and then start a thread that grabs the writeLock and
        // updates testLockValue3. We then assert that while we hold the readLock, the value is not updated. After we
        // release the readLock, the value should be updated.
        NativeMemoryAllocation.TrainingDataAllocation trainingDataAllocation = new NativeMemoryAllocation.TrainingDataAllocation(
            null,
            0,
            0,
            VectorDataType.FLOAT
        );

        int initialValue = 10;
        int finalValue = 16;
        testLockValue3 = initialValue;

        trainingDataAllocation.readLock();

        Thread thread = new Thread(() -> {
            trainingDataAllocation.writeLock();
            testLockValue3 = finalValue;
            trainingDataAllocation.writeUnlock();
        });

        thread.start();
        Thread.sleep(1000);

        assertEquals(initialValue, testLockValue3);
        trainingDataAllocation.readUnlock();

        Thread.sleep(1000);
        assertEquals(finalValue, testLockValue3);
    }

    public void testTrainingDataAllocation_writeLock() throws InterruptedException {
        // For trainingDataAllocations, the writeLock can be obtained in 1 thread and released in the other. In order to
        // test this, we grab the write lock in the initial thread, start a thread that tries to grab the readlock and
        // asserts that testLockValue4 is set to finalValue and then start another thread that updates testLockValue4
        // and releases the writeLock.
        NativeMemoryAllocation.TrainingDataAllocation trainingDataAllocation = new NativeMemoryAllocation.TrainingDataAllocation(
            null,
            0,
            0,
            VectorDataType.FLOAT
        );

        int initialValue = 10;
        int finalValue = 16;
        testLockValue4 = initialValue;

        trainingDataAllocation.writeLock();

        Thread thread1 = new Thread(() -> {
            testLockValue4 = finalValue;
            trainingDataAllocation.writeUnlock();
        });

        Thread thread2 = new Thread(() -> {
            trainingDataAllocation.readLock();
            assertEquals(finalValue, testLockValue4);
            trainingDataAllocation.readUnlock();
        });

        thread2.start();

        Thread.sleep(1000);

        thread1.start();

        Thread.sleep(1000);
    }

    public void testTrainingDataAllocation_getSize() {
        int size = 12;

        NativeMemoryAllocation.TrainingDataAllocation trainingDataAllocation = new NativeMemoryAllocation.TrainingDataAllocation(
            null,
            0,
            size,
            VectorDataType.FLOAT
        );

        assertEquals(size, trainingDataAllocation.getSizeInKB());
    }

    public void testTrainingDataAllocation_setMemoryAddress() {
        long pointer = 12;

        NativeMemoryAllocation.TrainingDataAllocation trainingDataAllocation = new NativeMemoryAllocation.TrainingDataAllocation(
            null,
            pointer,
            0,
            VectorDataType.FLOAT
        );

        assertEquals(pointer, trainingDataAllocation.getMemoryAddress());

        long newPointer = 18;
        trainingDataAllocation.setMemoryAddress(newPointer);
        assertEquals(newPointer, trainingDataAllocation.getMemoryAddress());
    }

    public void testAnonymousAllocation_close() throws InterruptedException {
        ExecutorService executorService = Executors.newSingleThreadExecutor();

        NativeMemoryAllocation.AnonymousAllocation anonymousAllocation = new NativeMemoryAllocation.AnonymousAllocation(executorService, 0);

        anonymousAllocation.close();

        Thread.sleep(1000 * 2);
        anonymousAllocation.writeLock();
        assertTrue(anonymousAllocation.isClosed());
        anonymousAllocation.writeUnlock();

        executorService.shutdown();
    }

    public void testAnonymousAllocation_getSize() {
        int size = 12;
        NativeMemoryAllocation.AnonymousAllocation anonymousAllocation = new NativeMemoryAllocation.AnonymousAllocation(null, size);

        assertEquals(size, anonymousAllocation.getSizeInKB());
    }
}

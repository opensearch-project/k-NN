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
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.junit.Before;
import org.mockito.Mock;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.jni.JNICommons;
import org.opensearch.knn.jni.JNIService;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicReference;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.featureflags.KNNFeatureFlags.KNN_FORCE_EVICT_CACHE_ENABLED_SETTING;

public class NativeMemoryAllocationTests extends KNNTestCase {

    private int testLockValue1;
    private int testLockValue2;
    private int testLockValue3;
    private int testLockValue4;

    @Mock
    ClusterSettings clusterSettings;

    @Before
    @Override
    public void setUp() throws Exception {
        super.setUp();
        clusterSettings = mock(ClusterSettings.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterSettings.get(KNN_FORCE_EVICT_CACHE_ENABLED_SETTING)).thenReturn(false);
        KNNSettings.state().setClusterService(clusterService);
    }

    @SneakyThrows
    public void testIndexAllocation_close() {
        // Create basic nmslib HNSW index
        Path tempDirPath = createTempDir();
        try (Directory directory = newFSDirectory(tempDirPath)) {
            KNNEngine knnEngine = KNNEngine.NMSLIB;
            String indexFileName = "test1" + knnEngine.getExtension();
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
            TestUtils.createIndex(ids, vectorMemoryAddress, dimension, directory, indexFileName, parameters, knnEngine);

            // Load index into memory
            final long memoryAddress;
            try (IndexInput indexInput = directory.openInput(indexFileName, IOContext.DEFAULT)) {
                final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(indexInput);
                memoryAddress = JNIService.loadIndex(indexInputWithBuffer, parameters, knnEngine);
            }

            ExecutorService executorService = Executors.newSingleThreadExecutor();
            NativeMemoryAllocation.IndexAllocation indexAllocation = new NativeMemoryAllocation.IndexAllocation(
                executorService,
                memoryAddress,
                (int) directory.fileLength(indexFileName) / 1024,
                knnEngine,
                indexFileName,
                "test"
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
    }

    @SneakyThrows
    public void testClose_whenBinaryFiass_thenSuccess() {
        Path tempDirPath = createTempDir();
        KNNEngine knnEngine = KNNEngine.FAISS;
        String indexFileName = "test1" + knnEngine.getExtension();
        try (Directory directory = newFSDirectory(tempDirPath)) {
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
            long vectorMemoryAddress = JNICommons.storeBinaryVectorData(0, vectors, numVectors * dataLength);
            TestUtils.createIndex(ids, vectorMemoryAddress, dimension, directory, indexFileName, parameters, knnEngine);

            // Load index into memory
            final long memoryAddress;
            try (IndexInput indexInput = directory.openInput(indexFileName, IOContext.DEFAULT)) {
                final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(indexInput);
                memoryAddress = JNIService.loadIndex(indexInputWithBuffer, parameters, knnEngine);
            }

            ExecutorService executorService = Executors.newSingleThreadExecutor();
            NativeMemoryAllocation.IndexAllocation indexAllocation = new NativeMemoryAllocation.IndexAllocation(
                executorService,
                memoryAddress,
                (int) directory.fileLength(indexFileName) / 1024,
                knnEngine,
                indexFileName,
                "test",
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
    }

    public void testIndexAllocation_getMemoryAddress() {
        long memoryAddress = 12;
        NativeMemoryAllocation.IndexAllocation indexAllocation = new NativeMemoryAllocation.IndexAllocation(
            null,
            memoryAddress,
            0,
            null,
            "test",
            "test"
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
            "test"
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

    public void testIndexAllocation_closeDefault() {
        ExecutorService executorService = Executors.newFixedThreadPool(2);
        AtomicReference<Exception> expectedException = new AtomicReference<>();

        // Executor based non-blocking close
        NativeMemoryAllocation.IndexAllocation nonBlockingIndexAllocation = new NativeMemoryAllocation.IndexAllocation(
            mock(ExecutorService.class),
            0,
            0,
            null,
            "test",
            "test"
        );

        executorService.submit(nonBlockingIndexAllocation::readLock);
        Future<?> closingThread = executorService.submit(nonBlockingIndexAllocation::close);
        try {
            closingThread.get();
        } catch (Exception ex) {
            expectedException.set(ex);
        }
        assertNull(expectedException.get());
        expectedException.set(null);
        executorService.shutdown();
    }

    public void testIndexAllocation_closeBlocking() throws InterruptedException, ExecutionException {
        // Prepare mocking and a thread pool.
        ExecutorService executorService = Executors.newSingleThreadExecutor();

        // Enable `KNN_FORCE_EVICT_CACHE_ENABLED_SETTING` to force it to block other threads.
        // Having it false will make `IndexAllocation` to run close logic in a different thread.
        when(clusterSettings.get(KNN_FORCE_EVICT_CACHE_ENABLED_SETTING)).thenReturn(true);
        NativeMemoryAllocation.IndexAllocation blockingIndexAllocation = new NativeMemoryAllocation.IndexAllocation(
            mock(ExecutorService.class),
            0,
            0,
            null,
            "test",
            "test"
        );

        // Acquire a read lock
        blockingIndexAllocation.readLock();

        // This should be blocked as a read lock is still being held.
        Future<?> closingThread = executorService.submit(blockingIndexAllocation::close);

        // Check if thread is currently blocked
        try {
            closingThread.get(5, TimeUnit.SECONDS);
            fail("Closing should be blocked. We are still holding a read lock.");
        } catch (TimeoutException ignored) {}

        // Now, we unlock a read lock.
        blockingIndexAllocation.readUnlock();
        // As we don't hold any locking, the closing thread can now good to acquire a write lock.
        closingThread.get();

        // Waits until close
        assertTrue(blockingIndexAllocation.isClosed());
        executorService.shutdown();
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
            "test"
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
            "test"
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
            "test"
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
            "test"
        );

        assertEquals(indexPath, indexAllocation.getVectorFileName());
    }

    public void testIndexAllocation_getOsIndexName() {
        String osIndexName = "test-index";
        NativeMemoryAllocation.IndexAllocation indexAllocation = new NativeMemoryAllocation.IndexAllocation(
            null,
            0,
            0,
            null,
            "test",
            osIndexName
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
        long memoryAddress = JNICommons.storeVectorData(0, vectors, vectors.length * dimension);

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

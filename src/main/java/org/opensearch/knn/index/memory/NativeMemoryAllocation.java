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

import org.apache.lucene.index.LeafReaderContext;
import org.opensearch.knn.index.JNIService;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.watcher.FileWatcher;
import org.opensearch.watcher.WatcherHandle;

import java.util.concurrent.Semaphore;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Represents a persistent allocation made in native memory. In this case, persistent means that the allocation is made
 * and not freed in the same call to the JNI. Therefore, in order to prevent memory leaks, we need to ensure that each
 * allocation is properly freed
 */
public interface NativeMemoryAllocation {

    /**
     * Closes the native memory allocation. It should deallocate all native memory associated with this allocation.
     */
    void close();

    /**
     * Check if the allocation has been closed.
     *
     * @return true if allocation has been closed; false otherwise
     */
    boolean isClosed();

    /**
     * Get the native memory pointer associated with the native memory allocation.
     *
     * @return pointer to native memory allocation
     */
    long getPointer();

    /**
     * Locks allocation for read. Multiple threads can obtain this lock assuming that no threads have the write lock.
     */
    void readLock();

    /**
     * Locks allocation for write. Only one thread can obtain this lock and no threads can have a read lock.
     */
    void writeLock();

    /**
     * Unlocks allocation for read.
     */
    void readUnlock();

    /**
     * Unlocks allocation for write.
     */
    void writeUnlock();

    /**
     * Get the size of the native memory allocation in kilobytes.
     *
     * @return size of native memory allocation
     */
    long getSizeInKb();

    /**
     * Represents native indices loaded into memory. Because these indices are backed by files, they should be
     * freed when file is deleted.
     */
    class IndexAllocation implements NativeMemoryAllocation {

        private final long pointer;
        private final long size;
        private volatile boolean closed;
        private final KNNEngine knnEngine;
        private final String indexPath;
        private final String openSearchIndexName;
        private final ReadWriteLock readWriteLock;
        private final WatcherHandle<FileWatcher> watcherHandle;

        /**
         * Constructor
         *
         * @param pointer Pointer in memory to the index
         * @param size Size this index consumes in kilobytes
         * @param knnEngine KNNEngine associated with the index allocation
         * @param indexPath File path to index
         * @param openSearchIndexName Name of OpenSearch index this index is associated with
         * @param watcherHandle Handle for watching index file
         */
        IndexAllocation(long pointer, long size, KNNEngine knnEngine, String indexPath,
                               String openSearchIndexName, WatcherHandle<FileWatcher> watcherHandle) {
            this.closed = false;
            this.knnEngine = knnEngine;
            this.indexPath = indexPath;
            this.openSearchIndexName = openSearchIndexName;
            this.pointer = pointer;
            this.readWriteLock = new ReentrantReadWriteLock();
            this.size = size;
            this.watcherHandle = watcherHandle;
        }

        @Override
        public void close() {
            // Lock acquisition should be done by caller
            if (this.closed) {
                return;
            }

            this.closed = true;

            watcherHandle.stop();

            // Pointer is sometimes initialized to 0. If this is ever the case, freeing will surely fail.
            if (pointer != 0) {
                JNIService.free(pointer, knnEngine.getName());
            }
        }

        @Override
        public boolean isClosed() {
            return closed;
        }

        @Override
        public long getPointer() {
            return pointer;
        }

        /**
         * The read lock will be obtained in the
         * {@link org.opensearch.knn.index.KNNWeight#scorer(LeafReaderContext context) scorer} when a native index needs
         * to be queried.
         */
        @Override
        public void readLock() {
            readWriteLock.readLock().lock();
        }

        /**
         * The write lock will be obtained in the
         * {@link NativeMemoryCacheManager NativeMemoryManager's} onRemoval function when the Index Allocation is
         * evicted from the cache. This prevents memory from being deallocated when it is being actively searched.
         */
        @Override
        public void writeLock() {
            readWriteLock.writeLock().lock();
        }

        @Override
        public void readUnlock() {
            readWriteLock.readLock().unlock();
        }

        @Override
        public void writeUnlock() {
            readWriteLock.writeLock().unlock();
        }

        @Override
        public long getSizeInKb() {
            return size;
        }

        /**
         * Getter for k-NN Engine associated with this index allocation.
         *
         * @return KNNEngine associated with index allocation
         */
        public KNNEngine getKnnEngine() {
            return knnEngine;
        }

        /**
         * Getter for the path to the file from which the index was loaded.
         *
         * @return indexPath to index
         */
        public String getIndexPath() {
            return indexPath;
        }

        /**
         * Getter for the OpenSearch index associated with the native index.
         *
         * @return OpenSearch index name
         */
        public String getOpenSearchIndexName() {
            return openSearchIndexName;
        }
    }

    /**
     * Represents training data that has been allocated in native memory.
     */
    class TrainingDataAllocation implements NativeMemoryAllocation {

        private volatile boolean closed;
        private long pointer;
        private final long size;

        // Implement reader/writer with semaphores to deal with passing lock conditions between threads
        private int readCount;
        private Semaphore readSemaphore;
        private Semaphore writeSemaphore;

        /**
         * Constructor
         *
         * @param pointer pointer in memory to the training data allocation
         * @param size amount memory needed for allocation in kilobytes
         */
        TrainingDataAllocation(long pointer, long size) {
            this.closed = false;
            this.pointer = pointer;
            this.size = size;

            this.readCount = 0;
            this.readSemaphore = new Semaphore(1);
            this.writeSemaphore = new Semaphore(1);
        }

        @Override
        public void close() {
            if (closed) {
                return;
            }

            closed = true;
            if (this.pointer != 0) {
                JNIService.freeVectors(this.pointer);
            }
        }

        @Override
        public boolean isClosed() {
            return closed;
        }

        @Override
        public long getPointer() {
            return pointer;
        }

        /**
         * A read lock will be obtained when a training job needs access to the TrainingDataAllocation.
         * In the future, we may want to switch to tryAcquire functionality.
         */
        @Override
        public void readLock() {
            try {
                readSemaphore.acquire();
            } catch (InterruptedException ex) {
                throw new RuntimeException(ex);
            }


            if (readCount == 0) {
                try {
                    writeSemaphore.acquire();
                } catch (InterruptedException e) {
                    readSemaphore.release();
                    throw new RuntimeException(e);
                }
            }

            readCount++;
            readSemaphore.release();
        }

        /**
         * A write lock will be obtained either on eviction from {@link NativeMemoryCacheManager NativeMemoryManager's}
         * or when training data is actually being loaded. A semaphore is used because collecting training data
         * happens asynchrously, so the thread that obtains the lock will not be the same thread that releases the
         * lock.
         */
        @Override
        public void writeLock() {
            try {
                writeSemaphore.acquire();
            } catch (InterruptedException ex) {
                throw new RuntimeException(ex);
            }
        }

        @Override
        public void readUnlock() {
            try {
                readSemaphore.acquire();
            } catch (InterruptedException ex) {
                throw new RuntimeException(ex);
            }

            readCount--;

            if (readCount == 0) {
                writeSemaphore.release();
            }

            readSemaphore.release();
        }

        @Override
        public void writeUnlock() {
            writeSemaphore.release();
        }

        @Override
        public long getSizeInKb() {
            return size;
        }

        /**
         * Setter for pointer to training data
         *
         * @param pointer Pointer to training data
         */
        public void setPointer(long pointer) {
            this.pointer = pointer;
        }
    }
}

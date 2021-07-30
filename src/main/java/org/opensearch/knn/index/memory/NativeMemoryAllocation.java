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
     * Locks allocation for read.
     */
    void readLock();

    /**
     * Locks allocation for write.
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
        private final String osIndexName;
        private final ReadWriteLock readWriteLock;
        private final WatcherHandle<FileWatcher> watcherHandle;

        /**
         * Constructor
         *
         * @param pointer Pointer in memory to the index
         * @param size Size this index consumes in kilobytes
         * @param knnEngine KNNEngine associated with the index allocation
         * @param indexPath File path to index
         * @param osIndexName Name of OpenSearch index this index is associated with
         * @param watcherHandle Handle for watching index file
         */
        public IndexAllocation(long pointer, long size, KNNEngine knnEngine, String indexPath, String osIndexName,
                               WatcherHandle<FileWatcher> watcherHandle) {
            this.closed = false;
            this.knnEngine = knnEngine;
            this.indexPath = indexPath;
            this.osIndexName = osIndexName;
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

            try {
                watcherHandle.stop();

                // Pointer is sometimes initialized to 0. If this is ever the case, freeing will surely fail.
                if (pointer != 0) {
                    JNIService.free(pointer, knnEngine.getName());
                }
            } finally {
                this.closed = true;
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

        @Override
        public void readLock() {
            readWriteLock.readLock().lock();
        }

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
        public String getOsIndexName() {
            return osIndexName;
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
        private Semaphore wrtieSemaphore;

        /**
         * Constructor
         *
         * @param pointer pointer in memory to the training data allocation
         * @param size amount memory needed for allocation in kilobytes
         */
        public TrainingDataAllocation(long pointer, long size) {
            this.closed = false;
            this.pointer = pointer;
            this.size = size;

            this.readCount = 0;
            this.readSemaphore = new Semaphore(1);
            this.wrtieSemaphore = new Semaphore(1);
        }

        @Override
        public void close() {
            if (closed) {
                return;
            }

            try {
                if (this.pointer != 0) {
                    JNIService.freeVectors(this.pointer);
                }
            } finally {
                closed = true;
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

        @Override
        public void readLock() {
            try {
                readSemaphore.acquire();
            } catch (InterruptedException ex) {
                throw new RuntimeException(ex);
            }


            if (readCount == 0) {
                try {
                    wrtieSemaphore.acquire();
                } catch (InterruptedException e) {
                    readSemaphore.release();
                    throw new RuntimeException(e);
                }
            }

            readCount++;
            readSemaphore.release();
        }

        @Override
        public void writeLock() {
            try {
                wrtieSemaphore.acquire();
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
                wrtieSemaphore.release();
            }

            readSemaphore.release();
        }

        @Override
        public void writeUnlock() {
            wrtieSemaphore.release();
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

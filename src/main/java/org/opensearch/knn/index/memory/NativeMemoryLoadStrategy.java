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

import org.opensearch.action.ActionListener;
import org.opensearch.knn.index.JNIService;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.training.TrainingDataConsumer;
import org.opensearch.knn.training.VectorReader;
import org.opensearch.watcher.FileChangesListener;
import org.opensearch.watcher.FileWatcher;
import org.opensearch.watcher.ResourceWatcherService;
import org.opensearch.watcher.WatcherHandle;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Responsible for loading entries from native memory.
 */
public interface NativeMemoryLoadStrategy<T extends NativeMemoryAllocation, U extends NativeMemoryEntryContext<T>> {

    /**
     * Loads a NativeMemoryAllocation from a given NativeMemoryEntryContext
     *
     * @param nativeMemoryEntryContext Context used to load nativeMemoryAllocation
     * @return Loaded NativeMemoryAllocation
     */
    T load(U nativeMemoryEntryContext) throws IOException;

    class IndexLoadStrategy implements NativeMemoryLoadStrategy<NativeMemoryAllocation.IndexAllocation,
            NativeMemoryEntryContext.IndexEntryContext> {

        private static IndexLoadStrategy INSTANCE;

        private final FileChangesListener indexFileOnDeleteListener;
        private ResourceWatcherService resourceWatcherService;

        /**
         * Get Singleton of this load strategy.
         *
         * @return singleton IndexLoadStrategy
         */
        public static synchronized IndexLoadStrategy getInstance() {
            if (INSTANCE == null) {
                INSTANCE = new IndexLoadStrategy();
            }
            return INSTANCE;
        }

        /**
         * Initialize singleton.
         *
         * @param resourceWatcherService service used to monitor index files for deletion
         */
        public static void initialize(final ResourceWatcherService resourceWatcherService) {
            getInstance().resourceWatcherService = resourceWatcherService;
        }

        private IndexLoadStrategy() {
            indexFileOnDeleteListener = new FileChangesListener() {
                @Override
                public void onFileDeleted(Path indexFilePath) {
                    NativeMemoryCacheManager.getInstance().invalidate(indexFilePath.toString());
                }
            };
        }

        @Override
        public NativeMemoryAllocation.IndexAllocation load(NativeMemoryEntryContext.IndexEntryContext
                                                                           indexEntryContext) throws IOException {
            Path indexPath = Paths.get(indexEntryContext.getKey());
            FileWatcher fileWatcher = new FileWatcher(indexPath);
            fileWatcher.addListener(indexFileOnDeleteListener);
            fileWatcher.init();

            KNNEngine knnEngine = KNNEngine.getEngineNameFromPath(indexPath.toString());
            long pointer = JNIService.loadIndex(indexPath.toString(), indexEntryContext.getParameters(),
                    knnEngine.getName());
            final WatcherHandle<FileWatcher> watcherHandle = resourceWatcherService.add(fileWatcher);

            return new NativeMemoryAllocation.IndexAllocation(
                    pointer,
                    indexEntryContext.calculateSizeInKb(),
                    knnEngine,
                    indexPath.toString(),
                    indexEntryContext.getOpenSearchIndexName(),
                    watcherHandle);
        }
    }

    class TrainingLoadStrategy implements NativeMemoryLoadStrategy<NativeMemoryAllocation.TrainingDataAllocation,
            NativeMemoryEntryContext.TrainingDataEntryContext> {

        private static TrainingLoadStrategy INSTANCE;
        private VectorReader vectorReader;

        /**
         * Get singleton TrainingLoadStrategy
         *
         * @return instance of TrainingLoadStrategy
         */
        public static synchronized TrainingLoadStrategy getInstance() {
            if (INSTANCE == null) {
                INSTANCE = new TrainingLoadStrategy();
            }
            return INSTANCE;
        }

        /**
         * Initialize singleton.
         *
         * @param vectorReader VectorReader used to read training data
         */
        public static void initialize(final VectorReader vectorReader) {
            getInstance().vectorReader = vectorReader;
        }

        private TrainingLoadStrategy() {}

        @Override
        public NativeMemoryAllocation.TrainingDataAllocation load(NativeMemoryEntryContext.TrainingDataEntryContext
                                                                                  nativeMemoryEntryContext) {
            // Generate an empty training data allocation with the appropriate size
            NativeMemoryAllocation.TrainingDataAllocation trainingDataAllocation = new NativeMemoryAllocation
                    .TrainingDataAllocation(0, nativeMemoryEntryContext.calculateSizeInKb());

            // Start loading all training data. Once the data has been loaded, release the lock
            TrainingDataConsumer trainingDataConsumer = new TrainingDataConsumer(trainingDataAllocation);

            trainingDataAllocation.writeLock();

            vectorReader.read(
                    nativeMemoryEntryContext.getIndicesService(),
                    nativeMemoryEntryContext.getTrainIndexName(),
                    nativeMemoryEntryContext.getTrainFieldName(),
                    nativeMemoryEntryContext.getMaxVectorCount(),
                    nativeMemoryEntryContext.getSearchSize(),
                    trainingDataConsumer,
                    ActionListener.wrap(
                            response -> trainingDataAllocation.writeUnlock(),
                            ex -> {
                                trainingDataAllocation.close();
                                trainingDataAllocation.writeUnlock();
                                throw new RuntimeException(ex);
                            }
                    )
            );


            // The write lock is acquired before the trainingDataAllocation is returned and not released until the
            // loading has completed. The calling thread will need to obtain a read lock in order to proceed, which
            // will not be possible until the write lock is released.
            return trainingDataAllocation;
        }
    }
}

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

        public static IndexLoadStrategy INSTANCE = new IndexLoadStrategy();

        private final FileChangesListener indexFileOnDeleteListener;
        private ResourceWatcherService resourceWatcherService;

        public static void initialize(final ResourceWatcherService resourceWatcherService) {
            INSTANCE.resourceWatcherService = resourceWatcherService;
        }

        /**
         * Empty Constructor
         */
        public IndexLoadStrategy() {
            indexFileOnDeleteListener = new FileChangesListener() {
                @Override
                public void onFileDeleted(Path indexFilePath) {
                    NativeMemoryCacheManager.getInstance().invalidate(indexFilePath.toString());
                }
            };
        }

        @Override
        public NativeMemoryAllocation.IndexAllocation load(NativeMemoryEntryContext.IndexEntryContext
                                                                           nativeMemoryEntryContext) throws IOException {
            Path indexPath = Paths.get(nativeMemoryEntryContext.getKey());
            FileWatcher fileWatcher = new FileWatcher(indexPath);
            fileWatcher.addListener(indexFileOnDeleteListener);
            fileWatcher.init();

            KNNEngine knnEngine = KNNEngine.getEngineNameFromPath(indexPath.toString());
            long pointer = JNIService.loadIndex(indexPath.toString(), nativeMemoryEntryContext.getParameters(),
                    knnEngine.getName());
            final WatcherHandle<FileWatcher> watcherHandle = resourceWatcherService.add(fileWatcher);

            return new NativeMemoryAllocation.IndexAllocation(pointer, nativeMemoryEntryContext.getSize(), knnEngine,
                    indexPath.toString(), nativeMemoryEntryContext.getOpenSearchIndexName(), watcherHandle);
        }
    }

    class TrainingLoadStrategy implements NativeMemoryLoadStrategy<NativeMemoryAllocation.TrainingDataAllocation,
            NativeMemoryEntryContext.TrainingDataEntryContext> {

        public static TrainingLoadStrategy INSTANCE = new TrainingLoadStrategy();
        private VectorReader vectorReader;

        public static void initialize(final VectorReader vectorReader) {
            INSTANCE.vectorReader = vectorReader;
        }

        /**
         * Empty Constructor
         */
        public TrainingLoadStrategy() {}

        @Override
        public NativeMemoryAllocation.TrainingDataAllocation load(NativeMemoryEntryContext.TrainingDataEntryContext
                                                                                  nativeMemoryEntryContext) {
            // Generate an empty training data allocation with the appropriate size
            NativeMemoryAllocation.TrainingDataAllocation trainingDataAllocation = new NativeMemoryAllocation
                    .TrainingDataAllocation(0, nativeMemoryEntryContext.getSize());

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
                                trainingDataAllocation.writeUnlock();
                                throw new RuntimeException(ex);
                            }
                    )
            );


            // Return the allocation immediately. The calling thread wont be able to do anything until the
            // vectors have all been loaded into native memory and the write lock has been released
            return trainingDataAllocation;
        }
    }
}

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

import lombok.extern.log4j.Log4j2;
import org.opensearch.core.action.ActionListener;
import org.opensearch.knn.index.IndexUtil;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.training.ByteTrainingDataConsumer;
import org.opensearch.knn.training.FloatTrainingDataConsumer;
import org.opensearch.knn.training.TrainingDataConsumer;
import org.opensearch.knn.training.VectorReader;
import org.opensearch.watcher.FileChangesListener;
import org.opensearch.watcher.FileWatcher;
import org.opensearch.watcher.ResourceWatcherService;
import org.opensearch.watcher.WatcherHandle;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

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

    @Log4j2
    class IndexLoadStrategy
        implements
            NativeMemoryLoadStrategy<NativeMemoryAllocation.IndexAllocation, NativeMemoryEntryContext.IndexEntryContext>,
            Closeable {

        private static IndexLoadStrategy INSTANCE;

        private final ExecutorService executor;
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
            executor = Executors.newSingleThreadExecutor();
            indexFileOnDeleteListener = new FileChangesListener() {
                @Override
                public void onFileDeleted(Path indexFilePath) {
                    NativeMemoryCacheManager.getInstance().invalidate(indexFilePath.toString());
                }
            };
        }

        @Override
        public NativeMemoryAllocation.IndexAllocation load(NativeMemoryEntryContext.IndexEntryContext indexEntryContext)
            throws IOException {
            Path indexPath = Paths.get(indexEntryContext.getKey());
            FileWatcher fileWatcher = new FileWatcher(indexPath);
            fileWatcher.addListener(indexFileOnDeleteListener);
            fileWatcher.init();

            KNNEngine knnEngine = KNNEngine.getEngineNameFromPath(indexPath.toString());
            long indexAddress = JNIService.loadIndex(indexPath.toString(), indexEntryContext.getParameters(), knnEngine);
            SharedIndexState sharedIndexState = null;
            String modelId = indexEntryContext.getModelId();
            if (IndexUtil.isSharedIndexStateRequired(knnEngine, modelId, indexAddress)) {
                log.info("Index with model: \"{}\" requires shared state. Retrieving shared state.", modelId);
                sharedIndexState = SharedIndexStateManager.getInstance().get(indexAddress, modelId, knnEngine);
                JNIService.setSharedIndexState(indexAddress, sharedIndexState.getSharedIndexStateAddress(), knnEngine);
            }

            final WatcherHandle<FileWatcher> watcherHandle = resourceWatcherService.add(fileWatcher);
            return new NativeMemoryAllocation.IndexAllocation(
                executor,
                indexAddress,
                indexEntryContext.calculateSizeInKB(),
                knnEngine,
                indexPath.toString(),
                indexEntryContext.getOpenSearchIndexName(),
                watcherHandle,
                sharedIndexState,
                IndexUtil.isBinaryIndex(knnEngine, indexEntryContext.getParameters())
            );
        }

        @Override
        public void close() {
            executor.shutdown();
        }
    }

    class TrainingLoadStrategy
        implements
            NativeMemoryLoadStrategy<NativeMemoryAllocation.TrainingDataAllocation, NativeMemoryEntryContext.TrainingDataEntryContext>,
            Closeable {

        private static TrainingLoadStrategy INSTANCE;

        private final ExecutorService executor;
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

        private TrainingLoadStrategy() {
            executor = Executors.newSingleThreadExecutor();
        }

        @Override
        public NativeMemoryAllocation.TrainingDataAllocation load(
            NativeMemoryEntryContext.TrainingDataEntryContext nativeMemoryEntryContext
        ) {
            // Generate an empty training data allocation with the appropriate size
            NativeMemoryAllocation.TrainingDataAllocation trainingDataAllocation = new NativeMemoryAllocation.TrainingDataAllocation(
                executor,
                0,
                nativeMemoryEntryContext.calculateSizeInKB(),
                nativeMemoryEntryContext.getVectorDataType()
            );

            TrainingDataConsumer vectorDataConsumer = nativeMemoryEntryContext.getVectorDataType() == VectorDataType.FLOAT
                ? new FloatTrainingDataConsumer(trainingDataAllocation)
                : new ByteTrainingDataConsumer(trainingDataAllocation);

            trainingDataAllocation.writeLock();

            vectorReader.read(
                nativeMemoryEntryContext.getClusterService(),
                nativeMemoryEntryContext.getTrainIndexName(),
                nativeMemoryEntryContext.getTrainFieldName(),
                nativeMemoryEntryContext.getMaxVectorCount(),
                nativeMemoryEntryContext.getSearchSize(),
                vectorDataConsumer,
                ActionListener.wrap(response -> trainingDataAllocation.writeUnlock(), ex -> {
                    // Close unsafe will assume that the caller passes control of the writelock to it. It
                    // will then handle releasing the write lock once the close operations finish.
                    trainingDataAllocation.closeUnsafe();
                    throw new RuntimeException(ex);
                })
            );

            // The write lock is acquired before the trainingDataAllocation is returned and not released until the
            // loading has completed. The calling thread will need to obtain a read lock in order to proceed, which
            // will not be possible until the write lock is released.
            return trainingDataAllocation;
        }

        @Override
        public void close() throws IOException {
            executor.shutdown();
        }
    }

    class AnonymousLoadStrategy
        implements
            NativeMemoryLoadStrategy<NativeMemoryAllocation.AnonymousAllocation, NativeMemoryEntryContext.AnonymousEntryContext>,
            Closeable {

        private static AnonymousLoadStrategy INSTANCE;

        /**
         * Get singleton AnonymousLoadStrategy
         *
         * @return instance of AnonymousLoadStrategy
         */
        public static synchronized AnonymousLoadStrategy getInstance() {
            if (INSTANCE == null) {
                INSTANCE = new AnonymousLoadStrategy();
            }
            return INSTANCE;
        }

        private final ExecutorService executor;

        private AnonymousLoadStrategy() {
            executor = Executors.newSingleThreadExecutor();
        }

        @Override
        public NativeMemoryAllocation.AnonymousAllocation load(NativeMemoryEntryContext.AnonymousEntryContext nativeMemoryEntryContext) {
            return new NativeMemoryAllocation.AnonymousAllocation(executor, nativeMemoryEntryContext.calculateSizeInKB());
        }

        @Override
        public void close() {
            executor.shutdown();
        }
    }
}

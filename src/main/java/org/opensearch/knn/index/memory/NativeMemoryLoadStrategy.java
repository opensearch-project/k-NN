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
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.opensearch.core.action.ActionListener;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.util.IndexUtil;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.training.TrainingDataConsumer;
import org.opensearch.knn.training.VectorReader;

import java.io.Closeable;
import java.io.IOException;
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

        private IndexLoadStrategy() {
            executor = Executors.newSingleThreadExecutor();
        }

        @Override
        public NativeMemoryAllocation.IndexAllocation load(NativeMemoryEntryContext.IndexEntryContext indexEntryContext)
            throws IOException {
            // Ex: _0_NativeEngines990KnnVectorsFormat_0.vec
            final String vectorFileName = indexEntryContext.getKey();
            final KNNEngine knnEngine = KNNEngine.getEngineNameFromPath(vectorFileName);

            final Directory directory = indexEntryContext.getDirectory();
            final int indexSizeKb = Math.toIntExact(directory.fileLength(vectorFileName) / 1024);

            try (IndexInput readStream = directory.openInput(vectorFileName, IOContext.READONCE)) {
                IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(readStream);
                long indexAddress = JNIService.loadIndex(indexInputWithBuffer, indexEntryContext.getParameters(), knnEngine);

                return createIndexAllocation(indexEntryContext, knnEngine, indexAddress, indexSizeKb, vectorFileName);
            }
        }

        private NativeMemoryAllocation.IndexAllocation createIndexAllocation(
            final NativeMemoryEntryContext.IndexEntryContext indexEntryContext,
            final KNNEngine knnEngine,
            final long indexAddress,
            final int indexSizeKb,
            final String vectorFileName
        ) {
            SharedIndexState sharedIndexState = null;
            String modelId = indexEntryContext.getModelId();
            if (IndexUtil.isSharedIndexStateRequired(knnEngine, modelId, indexAddress)) {
                log.info("Index with model: \"{}\" requires shared state. Retrieving shared state.", modelId);
                sharedIndexState = SharedIndexStateManager.getInstance().get(indexAddress, modelId, knnEngine);
                JNIService.setSharedIndexState(indexAddress, sharedIndexState.getSharedIndexStateAddress(), knnEngine);
            }

            return new NativeMemoryAllocation.IndexAllocation(
                executor,
                indexAddress,
                indexSizeKb,
                knnEngine,
                vectorFileName,
                indexEntryContext.getOpenSearchIndexName(),
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

        private static volatile TrainingLoadStrategy INSTANCE;

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

            QuantizationConfig quantizationConfig = nativeMemoryEntryContext.getQuantizationConfig();
            trainingDataAllocation.setQuantizationConfig(quantizationConfig);

            TrainingDataConsumer vectorDataConsumer = nativeMemoryEntryContext.getVectorDataType()
                .getTrainingDataConsumer(trainingDataAllocation);

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

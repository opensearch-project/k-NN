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

import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.codec.util.NativeMemoryCacheKeyHelper;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;
import java.util.Map;
import java.util.UUID;

/**
 * Encapsulates all information needed to load a component into native memory.
 */
public abstract class NativeMemoryEntryContext<T extends NativeMemoryAllocation> implements AutoCloseable {

    protected final String key;

    /**
     * Constructor
     *
     * @param key String used to identify entry in the cache
     */
    public NativeMemoryEntryContext(String key) {
        this.key = key;
    }

    /**
     * Getter for key.
     *
     * @return key
     */
    public String getKey() {
        return key;
    }

    /**
     * Calculate size for given context in kilobytes.
     *
     * @return size calculator
     */
    public abstract Integer calculateSizeInKB();

    /**
     * Opens the graph file by opening the corresponding indexInput so
     * that it is available for graph loading
     */

    public void open() {}

    /**
     * Provides the capability to close the closable objects in the {@link NativeMemoryEntryContext}
     */
    @Override
    public void close() {}

    /**
     * Loads entry into memory.
     *
     * @return NativeMemoryAllocation associated with NativeMemoryEntryContext
     */
    public abstract T load() throws IOException;

    @Log4j2
    public static class IndexEntryContext extends NativeMemoryEntryContext<NativeMemoryAllocation.IndexAllocation> {

        @Getter
        private final Directory directory;
        private final NativeMemoryLoadStrategy.IndexLoadStrategy indexLoadStrategy;
        @Getter
        private final String openSearchIndexName;
        @Getter
        private final Map<String, Object> parameters;
        @Nullable
        @Getter
        private final String modelId;

        @Getter
        private boolean indexGraphFileOpened = false;
        @Getter
        private int indexSizeKb;

        @Getter
        private IndexInput readStream;

        @Getter
        IndexInputWithBuffer indexInputWithBuffer;

        @Getter
        KNNVectorValues<?> knnVectorValues;

        /**
         * Constructor
         *
         * @param directory Lucene directory to create required IndexInput/IndexOutput to access files.
         * @param vectorIndexCacheKey Cache key for {@link NativeMemoryCacheManager}. It must contain a vector file name.
         * @param indexLoadStrategy Strategy to load index into memory
         * @param parameters Load time parameters
         * @param openSearchIndexName Opensearch index associated with index
         */
        public IndexEntryContext(
            Directory directory,
            String vectorIndexCacheKey,
            NativeMemoryLoadStrategy.IndexLoadStrategy indexLoadStrategy,
            Map<String, Object> parameters,
            String openSearchIndexName,
            KNNVectorValues<?> knnVectorValues
        ) {
            this(directory, vectorIndexCacheKey, indexLoadStrategy, parameters, openSearchIndexName, null, knnVectorValues);
        }

        /**
         * Constructor
         *
         * @param directory Lucene directory to create required IndexInput/IndexOutput to access files.
         * @param vectorIndexCacheKey Cache key for {@link NativeMemoryCacheManager}. It must contain a vector file name.
         * @param indexLoadStrategy strategy to load index into memory
         * @param parameters load time parameters
         * @param openSearchIndexName opensearch index associated with index
         * @param modelId model to be loaded. If none available, pass null
         */
        public IndexEntryContext(
            Directory directory,
            String vectorIndexCacheKey,
            NativeMemoryLoadStrategy.IndexLoadStrategy indexLoadStrategy,
            Map<String, Object> parameters,
            String openSearchIndexName,
            String modelId,
            KNNVectorValues<?> knnVectorValues
        ) {
            super(vectorIndexCacheKey);
            this.directory = directory;
            this.indexLoadStrategy = indexLoadStrategy;
            this.openSearchIndexName = openSearchIndexName;
            this.parameters = parameters;
            this.modelId = modelId;
            this.knnVectorValues = knnVectorValues;
        }

        @Override
        public Integer calculateSizeInKB() {
            final String indexFileName = NativeMemoryCacheKeyHelper.extractVectorIndexFileName(key);
            try {
                final long fileLength = directory.fileLength(indexFileName);
                return (int) (fileLength / 1024L);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public void open() {
            // if graph file is already opened for index, do nothing
            if (isIndexGraphFileOpened()) {
                return;
            }
            // Extract vector file name from the given cache key.
            // Ex: _0_165_my_field.faiss@1vaqiupVUwvkXAG4Qc/RPg==
            final String cacheKey = this.getKey();
            final String vectorFileName = NativeMemoryCacheKeyHelper.extractVectorIndexFileName(cacheKey);
            if (vectorFileName == null) {
                throw new IllegalStateException(
                    "Invalid cache key was given. The key [" + cacheKey + "] does not contain the corresponding vector file name."
                );
            }

            // Prepare for opening index input from directory.
            final Directory directory = this.getDirectory();

            // Try to open an index input then pass it down to native engine for loading an index.
            try {
                indexSizeKb = Math.toIntExact(directory.fileLength(vectorFileName) / 1024);
                readStream = directory.openInput(vectorFileName, IOContext.READONCE);
                readStream.seek(0);
                indexInputWithBuffer = new IndexInputWithBuffer(readStream);
                indexGraphFileOpened = true;
                log.debug("[KNN] NativeMemoryCacheManager open successful");
            } catch (IOException e) {
                throw new RuntimeException("Failed to open the index " + openSearchIndexName);
            }
        }

        @Override
        public NativeMemoryAllocation.IndexAllocation load() throws IOException {
            if (!isIndexGraphFileOpened()) {
                throw new IllegalStateException("Index graph file is not open");
            }
            return indexLoadStrategy.load(this);
        }

        // close the indexInput
        @Override
        public void close() {
            if (readStream != null) {
                try {
                    readStream.close();
                    indexGraphFileOpened = false;
                } catch (IOException e) {
                    throw new RuntimeException(
                        "Exception while closing the indexInput index [" + openSearchIndexName + "] for loading the graph file.",
                        e
                    );
                }
            }
        }
    }

    public static class TrainingDataEntryContext extends NativeMemoryEntryContext<NativeMemoryAllocation.TrainingDataAllocation> {

        private static final String KEY_PREFIX = "tdata#";
        private static final String DELIMETER = ":";

        private final int size;
        private final NativeMemoryLoadStrategy.TrainingLoadStrategy trainingLoadStrategy;
        private final ClusterService clusterService;
        private final String trainIndexName;
        private final String trainFieldName;
        private final int maxVectorCount;
        private final int searchSize;
        private final VectorDataType vectorDataType;
        @Getter
        private final QuantizationConfig quantizationConfig;

        /**
         * Constructor
         *
         * @param size amount of memory training data will occupy in kilobytes
         * @param trainIndexName name of index used to pull training data from
         * @param trainFieldName name of field used to pull training data from
         * @param trainingLoadStrategy strategy to load training data into memory
         * @param clusterService service used to extract information about indices
         * @param maxVectorCount maximum number of vectors there can be
         * @param searchSize size each search request should return during loading
         */
        public TrainingDataEntryContext(
            int size,
            String trainIndexName,
            String trainFieldName,
            NativeMemoryLoadStrategy.TrainingLoadStrategy trainingLoadStrategy,
            ClusterService clusterService,
            int maxVectorCount,
            int searchSize,
            VectorDataType vectorDataType,
            QuantizationConfig quantizationConfig
        ) {
            super(generateKey(trainIndexName, trainFieldName));
            this.size = size;
            this.trainingLoadStrategy = trainingLoadStrategy;
            this.trainIndexName = trainIndexName;
            this.trainFieldName = trainFieldName;
            this.clusterService = clusterService;
            this.maxVectorCount = maxVectorCount;
            this.searchSize = searchSize;
            this.vectorDataType = vectorDataType;
            this.quantizationConfig = quantizationConfig;
        }

        @Override
        public Integer calculateSizeInKB() {
            return size;
        }

        @Override
        public NativeMemoryAllocation.TrainingDataAllocation load() {
            return trainingLoadStrategy.load(this);
        }

        /**
         * Getter for training index name.
         *
         * @return train index name
         */
        public String getTrainIndexName() {
            return trainIndexName;
        }

        /**
         * Getter for training index field.
         *
         * @return train field name
         */
        public String getTrainFieldName() {
            return trainFieldName;
        }

        /**
         * Getter for maximum number of vectors.
         *
         * @return maximum number of vectors
         */
        public int getMaxVectorCount() {
            return maxVectorCount;
        }

        /**
         * Getter for search size.
         *
         * @return size of results each search should return
         */
        public int getSearchSize() {
            return searchSize;
        }

        /**
         * Getter for cluster service.
         *
         * @return cluster service
         */
        public ClusterService getClusterService() {
            return clusterService;
        }

        /**
         * Getter for vector data type.
         *
         * @return vector data type
         */
        public VectorDataType getVectorDataType() {
            return vectorDataType;
        }

        private static String generateKey(String trainIndexName, String trainFieldName) {
            return KEY_PREFIX + trainIndexName + DELIMETER + trainFieldName;
        }
    }

    public static class AnonymousEntryContext extends NativeMemoryEntryContext<NativeMemoryAllocation.AnonymousAllocation> {

        private final int size;
        private final NativeMemoryLoadStrategy.AnonymousLoadStrategy loadStrategy;

        /**
         * Constructor
         *
         * @param size Size of the entry
         * @param loadStrategy strategy to load anonymous allocation into memory
         */
        public AnonymousEntryContext(int size, NativeMemoryLoadStrategy.AnonymousLoadStrategy loadStrategy) {
            super(UUID.randomUUID().toString());
            this.size = size;
            this.loadStrategy = loadStrategy;
        }

        @Override
        public Integer calculateSizeInKB() {
            return size;
        }

        @Override
        public NativeMemoryAllocation.AnonymousAllocation load() throws IOException {
            return loadStrategy.load(this);
        }
    }
}

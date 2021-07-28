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

import org.opensearch.indices.IndicesService;
import org.opensearch.knn.index.IndexUtil;

import java.io.IOException;
import java.util.Map;
import java.util.function.Function;

/**
 * Encapsulates all information needed to load a component into native memory.
 */
public abstract class NativeMemoryEntryContext<T extends NativeMemoryAllocation> {

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
    public abstract Long calculateSizeInKb();

    /**
     * Loads entry into memory.
     *
     * @return NativeMemoryAllocation associated with NativeMemoryEntryContext
     */
    public abstract T load() throws IOException;

    public static class IndexEntryContext extends NativeMemoryEntryContext<NativeMemoryAllocation.IndexAllocation> {

        private final NativeMemoryLoadStrategy.IndexLoadStrategy indexLoadStrategy;
        private final String openSearchIndexName;
        private final Map<String, Object> parameters;

        /**
         * Constructor
         *
         * @param indexPath path to index file. Also used as key in cache.
         * @param indexLoadStrategy strategy to load index into memory
         * @param parameters load time parameters
         * @param openSearchIndexName opensearch index associated with index
         */
        public IndexEntryContext(String indexPath,
                                 NativeMemoryLoadStrategy.IndexLoadStrategy indexLoadStrategy,
                                 Map<String, Object> parameters,
                                 String openSearchIndexName) {
            super(indexPath);
            this.indexLoadStrategy = indexLoadStrategy;
            this.openSearchIndexName = openSearchIndexName;
            this.parameters = parameters;
        }

        @Override
        public Long calculateSizeInKb() {
            return IndexSizeCalculator.INSTANCE.apply(this);
        }

        @Override
        public NativeMemoryAllocation.IndexAllocation load() throws IOException {
            return indexLoadStrategy.load(this);
        }

        /**
         * Getter for OpenSearch index name.
         *
         * @return OpenSearch index name
         */
        public String getOpenSearchIndexName() {
            return openSearchIndexName;
        }

        /**
         * Getter for parameters.
         *
         * @return parameters
         */
        public Map<String, Object> getParameters() {
            return parameters;
        }

        private static class IndexSizeCalculator implements Function<IndexEntryContext, Long> {

            static IndexSizeCalculator INSTANCE = new IndexSizeCalculator();

            IndexSizeCalculator() {}

            @Override
            public Long apply(IndexEntryContext indexEntryContext) {
                return IndexUtil.getFileSizeInKB(indexEntryContext.getKey());
            }
        }
    }

    public static class TrainingDataEntryContext extends NativeMemoryEntryContext<NativeMemoryAllocation.TrainingDataAllocation> {

        private static String KEY_PREFIX = "tdata#";
        private static String DELIMETER = ":";

        private final long size;
        private final NativeMemoryLoadStrategy.TrainingLoadStrategy trainingLoadStrategy;
        private final IndicesService indicesService;
        private final String trainIndexName;
        private final String trainFieldName;
        private final int maxVectorCount;
        private final int searchSize;

        /**
         * Constructor
         *
         * @param size amount of memory training data will occupy in kilobytes
         * @param trainIndexName name of index used to pull training data from
         * @param trainFieldName name of field used to pull training data from
         * @param trainingLoadStrategy strategy to load training data into memory
         * @param indicesService service used to extract information about indices
         * @param maxVectorCount maximum number of vectors there can be
         * @param searchSize size each search request should return during loading
         */
        public TrainingDataEntryContext(long size,
                                        String trainIndexName,
                                        String trainFieldName,
                                        NativeMemoryLoadStrategy.TrainingLoadStrategy trainingLoadStrategy,
                                        IndicesService indicesService,
                                        int maxVectorCount,
                                        int searchSize) {
            super(generateKey(trainIndexName, trainFieldName));
            this.size = size;
            this.trainingLoadStrategy = trainingLoadStrategy;
            this.trainIndexName = trainIndexName;
            this.trainFieldName = trainFieldName;
            this.indicesService = indicesService;
            this.maxVectorCount = maxVectorCount;
            this.searchSize = searchSize;
        }

        @Override
        public Long calculateSizeInKb() {
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
         * Getter for indices service.
         *
         * @return indices service
         */
        public IndicesService getIndicesService() {
            return indicesService;
        }

        private static String generateKey(String trainIndexName, String trainFieldName) {
            return KEY_PREFIX + trainIndexName + DELIMETER + trainFieldName;
        }
    }
}

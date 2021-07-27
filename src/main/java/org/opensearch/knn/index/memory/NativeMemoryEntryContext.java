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
import org.opensearch.knn.index.SpaceType;

import java.io.IOException;
import java.util.Map;

/**
 * Encapsulates all information needed to load a component into native memory.
 */
public abstract class NativeMemoryEntryContext<T extends NativeMemoryAllocation> {

    protected final String key;
    protected final long size;

    /**
     * Constructor
     *
     * @param key String used to identify entry in the cache
     * @param size size this allocation will take up in the cache
     */
    public NativeMemoryEntryContext(String key, long size) {
        this.key = key;
        this.size = size;
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
     * Getter for size.
     *
     * @return size
     */
    public long getSize() {
        return size;
    }

    /**
     * Loads entry into memory.
     *
     * @return NativeMemoryAllocation associated with NativeMemoryEntryContext
     */
    public abstract T load() throws IOException;

    public static class IndexEntryContext extends NativeMemoryEntryContext<NativeMemoryAllocation.IndexAllocation> {

        private final NativeMemoryLoadStrategy.IndexLoadStrategy indexLoadStrategy;
        private final String openSearchIndexName;
        private final SpaceType spaceType;
        private final Map<String, Object> parameters;

        /**
         * Constructor
         *
         * @param indexPath path to index file. Also used as key in cache.
         * @param size amount of memory the index occupies
         * @param indexLoadStrategy strategy to load index into memory
         * @param parameters load time parameters
         * @param openSearchIndexName opensearch index associated with index
         * @param spaceType space this index uses
         */
        public IndexEntryContext(String indexPath,
                                 long size,
                                 NativeMemoryLoadStrategy.IndexLoadStrategy indexLoadStrategy,
                                 Map<String, Object> parameters,
                                 String openSearchIndexName,
                                 SpaceType spaceType) {
            super(indexPath, size);
            this.indexLoadStrategy = indexLoadStrategy;
            this.openSearchIndexName = openSearchIndexName;
            this.spaceType = spaceType;
            this.parameters = parameters;
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
         * Getter for space type.
         *
         * @return spaceType
         */
        public SpaceType getSpaceType() {
            return spaceType;
        }

        /**
         * Getter for parameters.
         *
         * @return parameters
         */
        public Map<String, Object> getParameters() {
            return parameters;
        }
    }

    public static class TrainingDataEntryContext extends NativeMemoryEntryContext<NativeMemoryAllocation.TrainingDataAllocation> {

        private static String KEY_PREFIX = "tdata#";
        private static String DELIMETER = ":";

        private final NativeMemoryLoadStrategy.TrainingLoadStrategy trainingLoadStrategy;
        private final IndicesService indicesService;
        private final String trainIndexName;
        private final String trainFieldName;
        private final int maxVectorCount;
        private final int searchSize;

        /**
         * Constructor
         *
         * @param size amount of memory training data will occupy
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
            super(generateKey(trainIndexName, trainFieldName), size);
            this.trainingLoadStrategy = trainingLoadStrategy;
            this.trainIndexName = trainIndexName;
            this.trainFieldName = trainFieldName;
            this.indicesService = indicesService;
            this.maxVectorCount = maxVectorCount;
            this.searchSize = searchSize;
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

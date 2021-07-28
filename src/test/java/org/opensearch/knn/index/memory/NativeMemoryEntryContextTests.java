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
import org.opensearch.indices.IndicesService;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;
import java.util.Map;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class NativeMemoryEntryContextTests extends KNNTestCase {

    public void testAbstract_getKey() {
        String key = "test-1";
        TestNativeMemoryEntryContext testNativeMemoryEntryContext = new TestNativeMemoryEntryContext(key, 10);

        assertEquals(key, testNativeMemoryEntryContext.getKey());
    }

    public void testAbstract_getSize() {
        long size = 10;
        TestNativeMemoryEntryContext testNativeMemoryEntryContext = new TestNativeMemoryEntryContext("test-1", size);

        assertEquals(size, testNativeMemoryEntryContext.getSize());
    }

    public void testIndexEntryContext_load() throws IOException {
        NativeMemoryLoadStrategy.IndexLoadStrategy indexLoadStrategy = mock(NativeMemoryLoadStrategy.IndexLoadStrategy.class);
        NativeMemoryEntryContext.IndexEntryContext indexEntryContext = new NativeMemoryEntryContext.IndexEntryContext(
                "test",
                0,
                indexLoadStrategy,
                null,
                "test",
                null
        );

        NativeMemoryAllocation.IndexAllocation indexAllocation = new NativeMemoryAllocation.IndexAllocation(
                0,
                10,
                KNNEngine.DEFAULT,
                "test-path",
                "test-name",
                null
        );

        when(indexLoadStrategy.load(indexEntryContext)).thenReturn(indexAllocation);

        assertEquals(indexAllocation, indexEntryContext.load());
    }

    public void testIndexEntryContext_getOpenSearchIndexName() {
        String openSearchIndexName = "test-index";
        NativeMemoryEntryContext.IndexEntryContext indexEntryContext = new NativeMemoryEntryContext.IndexEntryContext(
                "test",
                0,
                null,
                null,
                openSearchIndexName,
                null
        );

        assertEquals(openSearchIndexName, indexEntryContext.getOpenSearchIndexName());
    }

    public void testIndexEntryContext_getSpaceType() {
        SpaceType spaceType = SpaceType.L1;
        NativeMemoryEntryContext.IndexEntryContext indexEntryContext = new NativeMemoryEntryContext.IndexEntryContext(
                "test",
                0,
                null,
                null,
                "test",
                spaceType
        );

        assertEquals(spaceType, indexEntryContext.getSpaceType());
    }

    public void testIndexEntryContext_getParameters() {
        Map<String, Object> parameters = ImmutableMap.of("test-1", 10);
        NativeMemoryEntryContext.IndexEntryContext indexEntryContext = new NativeMemoryEntryContext.IndexEntryContext(
                "test",
                0,
                null,
                parameters,
                "test",
                null
        );

        assertEquals(parameters, indexEntryContext.getParameters());
    }

    public void testTrainingDataEntryContext_load() {
        NativeMemoryLoadStrategy.TrainingLoadStrategy trainingLoadStrategy = mock(NativeMemoryLoadStrategy.TrainingLoadStrategy.class);
        NativeMemoryEntryContext.TrainingDataEntryContext trainingDataEntryContext = new NativeMemoryEntryContext.TrainingDataEntryContext(
                0,
                "test",
                "test",
                trainingLoadStrategy,
                null,
                0,
                0
        );

        NativeMemoryAllocation.TrainingDataAllocation trainingDataAllocation = new NativeMemoryAllocation.TrainingDataAllocation(
                0,
                0
        );

        when(trainingLoadStrategy.load(trainingDataEntryContext)).thenReturn(trainingDataAllocation);

        assertEquals(trainingDataAllocation, trainingDataEntryContext.load());
    }

    public void testTrainingDataEntryContext_getTrainIndexName() {
        String trainIndexName = "test-index";
        NativeMemoryEntryContext.TrainingDataEntryContext trainingDataEntryContext = new NativeMemoryEntryContext.TrainingDataEntryContext(
                0,
                trainIndexName,
                "test",
                null,
                null,
                0,
                0
        );

        assertEquals(trainIndexName, trainingDataEntryContext.getTrainIndexName());
    }

    public void testTrainingDataEntryContext_getTrainFieldName() {
        String trainFieldName = "test-field";
        NativeMemoryEntryContext.TrainingDataEntryContext trainingDataEntryContext = new NativeMemoryEntryContext.TrainingDataEntryContext(
                0,
                "test",
                trainFieldName,
                null,
                null,
                0,
                0
        );

        assertEquals(trainFieldName, trainingDataEntryContext.getTrainFieldName());
    }

    public void testTrainingDataEntryContext_getMaxVectorCount() {
        int maxVectorCount = 11;
        NativeMemoryEntryContext.TrainingDataEntryContext trainingDataEntryContext = new NativeMemoryEntryContext.TrainingDataEntryContext(
                0,
                "test",
                "test",
                null,
                null,
                maxVectorCount,
                0
        );

        assertEquals(maxVectorCount, trainingDataEntryContext.getMaxVectorCount());
    }

    public void testTrainingDataEntryContext_getSearchSize() {
        int searchSize = 11;
        NativeMemoryEntryContext.TrainingDataEntryContext trainingDataEntryContext = new NativeMemoryEntryContext.TrainingDataEntryContext(
                0,
                "test",
                "test",
                null,
                null,
                0,
                searchSize
        );

        assertEquals(searchSize, trainingDataEntryContext.getSearchSize());
    }

    public void testTrainingDataEntryContext_getIndicesService() {
        IndicesService indicesService = mock(IndicesService.class);
        NativeMemoryEntryContext.TrainingDataEntryContext trainingDataEntryContext = new NativeMemoryEntryContext.TrainingDataEntryContext(
                0,
                "test",
                "test",
                null,
                indicesService,
                0,
                0
        );

        assertEquals(indicesService, trainingDataEntryContext.getIndicesService());
    }

    private static class TestNativeMemoryAllocation implements NativeMemoryAllocation {

        @Override
        public void close() {

        }

        @Override
        public boolean isClosed() {
            return false;
        }

        @Override
        public long getPointer() {
            return 0;
        }

        @Override
        public void readLock() {

        }

        @Override
        public void writeLock() {

        }

        @Override
        public void readUnlock() {

        }

        @Override
        public void writeUnlock() {

        }

        @Override
        public long getSize() {
            return 0;
        }
    }

    private static class TestNativeMemoryEntryContext extends NativeMemoryEntryContext<TestNativeMemoryAllocation> {

        /**
         * Constructor
         *
         * @param key  String used to identify entry in the cache
         * @param size size this allocation will take up in the cache
         */
        public TestNativeMemoryEntryContext(String key, long size) {
            super(key, size);
        }

        @Override
        public TestNativeMemoryAllocation load() throws IOException {
            return null;
        }
    }
}

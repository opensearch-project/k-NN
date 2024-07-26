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
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.IndexUtil;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;

import static java.nio.file.StandardOpenOption.APPEND;
import static java.nio.file.StandardOpenOption.CREATE;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class NativeMemoryEntryContextTests extends KNNTestCase {

    public void testAbstract_getKey() {
        String key = "test-1";
        TestNativeMemoryEntryContext testNativeMemoryEntryContext = new TestNativeMemoryEntryContext(key, 10);

        assertEquals(key, testNativeMemoryEntryContext.getKey());
    }

    public void testIndexEntryContext_load() throws IOException {
        NativeMemoryLoadStrategy.IndexLoadStrategy indexLoadStrategy = mock(NativeMemoryLoadStrategy.IndexLoadStrategy.class);
        NativeMemoryEntryContext.IndexEntryContext indexEntryContext = new NativeMemoryEntryContext.IndexEntryContext(
            "test",
            indexLoadStrategy,
            null,
            "test"
        );

        NativeMemoryAllocation.IndexAllocation indexAllocation = new NativeMemoryAllocation.IndexAllocation(
            null,
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

    public void testIndexEntryContext_calculateSize() throws IOException {
        // Create a file and write random bytes to it
        Path tmpFile = createTempFile();
        byte[] data = new byte[1024 * 3];
        Arrays.fill(data, (byte) 'c');

        try (OutputStream out = new BufferedOutputStream(Files.newOutputStream(tmpFile, CREATE, APPEND))) {
            out.write(data, 0, data.length);
        } catch (IOException x) {
            fail("Failed to write to file");
        }

        // Get the expected size of this function
        int expectedSize = IndexUtil.getFileSizeInKB(tmpFile.toAbsolutePath().toString());

        // Check that the indexEntryContext will return the same thing
        NativeMemoryEntryContext.IndexEntryContext indexEntryContext = new NativeMemoryEntryContext.IndexEntryContext(
            tmpFile.toAbsolutePath().toString(),
            null,
            null,
            "test"
        );

        assertEquals(expectedSize, indexEntryContext.calculateSizeInKB().longValue());
    }

    public void testIndexEntryContext_getOpenSearchIndexName() {
        String openSearchIndexName = "test-index";
        NativeMemoryEntryContext.IndexEntryContext indexEntryContext = new NativeMemoryEntryContext.IndexEntryContext(
            "test",
            null,
            null,
            openSearchIndexName
        );

        assertEquals(openSearchIndexName, indexEntryContext.getOpenSearchIndexName());
    }

    public void testIndexEntryContext_getParameters() {
        Map<String, Object> parameters = ImmutableMap.of("test-1", 10);
        NativeMemoryEntryContext.IndexEntryContext indexEntryContext = new NativeMemoryEntryContext.IndexEntryContext(
            "test",
            null,
            parameters,
            "test"
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
            0,
            VectorDataType.DEFAULT
        );

        NativeMemoryAllocation.TrainingDataAllocation trainingDataAllocation = new NativeMemoryAllocation.TrainingDataAllocation(
            null,
            0,
            0,
            VectorDataType.DEFAULT
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
            0,
            VectorDataType.DEFAULT
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
            0,
            VectorDataType.DEFAULT
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
            0,
            VectorDataType.DEFAULT
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
            searchSize,
            VectorDataType.DEFAULT
        );

        assertEquals(searchSize, trainingDataEntryContext.getSearchSize());
    }

    public void testTrainingDataEntryContext_getIndicesService() {
        ClusterService clusterService = mock(ClusterService.class);
        NativeMemoryEntryContext.TrainingDataEntryContext trainingDataEntryContext = new NativeMemoryEntryContext.TrainingDataEntryContext(
            0,
            "test",
            "test",
            null,
            clusterService,
            0,
            0,
            VectorDataType.DEFAULT
        );

        assertEquals(clusterService, trainingDataEntryContext.getClusterService());
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
        public long getMemoryAddress() {
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
        public int getSizeInKB() {
            return 0;
        }
    }

    private static class TestNativeMemoryEntryContext extends NativeMemoryEntryContext<TestNativeMemoryAllocation> {

        int size;

        /**
         * Constructor
         *
         * @param key  String used to identify entry in the cache
         * @param size size this allocation will take up in the cache
         */
        public TestNativeMemoryEntryContext(String key, int size) {
            super(key);
            this.size = size;
        }

        @Override
        public Integer calculateSizeInKB() {
            return size;
        }

        @Override
        public TestNativeMemoryAllocation load() throws IOException {
            return null;
        }
    }
}

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
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.doReturn;

public class NativeMemoryEntryContextTests extends KNNTestCase {

    public void testAbstract_getKey() {
        String key = "test-1";
        TestNativeMemoryEntryContext testNativeMemoryEntryContext = new TestNativeMemoryEntryContext(key, 10);

        assertEquals(key, testNativeMemoryEntryContext.getKey());
    }

    public void testIndexEntryContext_load() throws IOException {
        NativeMemoryLoadStrategy.IndexLoadStrategy indexLoadStrategy = mock(NativeMemoryLoadStrategy.IndexLoadStrategy.class);
        final KNNVectorValues<?> knnVectorValues = mock(KNNVectorValues.class);
        NativeMemoryEntryContext.IndexEntryContext indexEntryContext = spy(
            new NativeMemoryEntryContext.IndexEntryContext(
                (Directory) null,
                TestUtils.createFakeNativeMamoryCacheKey("test"),
                indexLoadStrategy,
                null,
                "test",
                knnVectorValues
            )
        );

        NativeMemoryAllocation.IndexAllocation indexAllocation = new NativeMemoryAllocation.IndexAllocation(
            null,
            0,
            10,
            KNNEngine.DEFAULT,
            "test-path",
            "test-name"
        );

        when(indexLoadStrategy.load(indexEntryContext)).thenReturn(indexAllocation);

        // since we are returning mock instance, set indexEntryContext.isIndexGraphFileOpened to true.
        doReturn(true).when(indexEntryContext).isIndexGraphFileOpened();
        assertEquals(indexAllocation, indexEntryContext.load());
    }

    public void testIndexEntryContext_load_with_unopened_graphFile() throws IOException {
        NativeMemoryLoadStrategy.IndexLoadStrategy indexLoadStrategy = mock(NativeMemoryLoadStrategy.IndexLoadStrategy.class);
        final KNNVectorValues<?> knnVectorValues = mock(KNNVectorValues.class);
        NativeMemoryEntryContext.IndexEntryContext indexEntryContext = new NativeMemoryEntryContext.IndexEntryContext(
            (Directory) null,
            TestUtils.createFakeNativeMamoryCacheKey("test"),
            indexLoadStrategy,
            null,
            "test",
            knnVectorValues
        );

        NativeMemoryAllocation.IndexAllocation indexAllocation = new NativeMemoryAllocation.IndexAllocation(
            null,
            0,
            10,
            KNNEngine.DEFAULT,
            "test-path",
            "test-name"
        );

        assertThrows(IllegalStateException.class, indexEntryContext::load);
    }

    public void testIndexEntryContext_calculateSize() throws IOException {
        // Create a file and write random bytes to it
        final Path tmpDirectory = createTempDir();
        final Directory directory = new MMapDirectory(tmpDirectory);
        final String indexFileName = "test.faiss";
        byte[] data = new byte[1024 * 3];
        Arrays.fill(data, (byte) 'c');

        try (IndexOutput output = directory.createOutput(indexFileName, IOContext.DEFAULT)) {
            output.writeBytes(data, data.length);
        }

        // Get the expected size of this function
        final long expectedSizeBytes = directory.fileLength(indexFileName);
        final long expectedSizeKb = expectedSizeBytes / 1024L;
        final KNNVectorValues<?> knnVectorValues = mock(KNNVectorValues.class);

        // Check that the indexEntryContext will return the same thing
        NativeMemoryEntryContext.IndexEntryContext indexEntryContext = new NativeMemoryEntryContext.IndexEntryContext(
            directory,
            TestUtils.createFakeNativeMamoryCacheKey(indexFileName),
            null,
            null,
            "test",
            knnVectorValues
        );

        assertEquals(expectedSizeKb, indexEntryContext.calculateSizeInKB().longValue());
    }

    public void testIndexEntryContext_getOpenSearchIndexName() {
        String openSearchIndexName = "test-index";
        final KNNVectorValues<?> knnVectorValues = mock(KNNVectorValues.class);
        NativeMemoryEntryContext.IndexEntryContext indexEntryContext = new NativeMemoryEntryContext.IndexEntryContext(
            (Directory) null,
            TestUtils.createFakeNativeMamoryCacheKey("test"),
            null,
            null,
            openSearchIndexName,
            knnVectorValues
        );

        assertEquals(openSearchIndexName, indexEntryContext.getOpenSearchIndexName());
    }

    public void testIndexEntryContext_getParameters() {
        Map<String, Object> parameters = ImmutableMap.of("test-1", 10);
        final KNNVectorValues<?> knnVectorValues = mock(KNNVectorValues.class);
        NativeMemoryEntryContext.IndexEntryContext indexEntryContext = new NativeMemoryEntryContext.IndexEntryContext(
            (Directory) null,
            TestUtils.createFakeNativeMamoryCacheKey("test"),
            null,
            parameters,
            "test",
            knnVectorValues
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
            VectorDataType.DEFAULT,
            QuantizationConfig.EMPTY
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
            VectorDataType.DEFAULT,
            QuantizationConfig.EMPTY
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
            VectorDataType.DEFAULT,
            QuantizationConfig.EMPTY
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
            VectorDataType.DEFAULT,
            QuantizationConfig.EMPTY
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
            VectorDataType.DEFAULT,
            QuantizationConfig.EMPTY
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
            VectorDataType.DEFAULT,
            QuantizationConfig.EMPTY
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
        public void open() {
            return;
        }

        @Override
        public TestNativeMemoryAllocation load() throws IOException {
            return null;
        }
    }
}

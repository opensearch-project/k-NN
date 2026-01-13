/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.warmup;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.AlreadyClosedException;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Version;
import org.junit.Before;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.Metadata;
import org.opensearch.common.settings.Settings;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.atLeastOnce;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.verify;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

public class MemoryOptimizedSearchWarmupTests extends KNNTestCase {

    private final String testDescription;
    private final CompressionLevel compressionLevel;
    private final QuantizationConfig quantizationConfig;

    public MemoryOptimizedSearchWarmupTests(
        String testDescription,
        CompressionLevel compressionLevel,
        QuantizationConfig quantizationConfig
    ) {
        this.testDescription = testDescription;
        this.compressionLevel = compressionLevel;
        this.quantizationConfig = quantizationConfig;
    }

    @ParametersFactory
    public static Collection<Object[]> parameters() {
        return Arrays.asList(
            new Object[][] {
                { "1x compression", CompressionLevel.x1, QuantizationConfig.EMPTY },
                { "2x compression", CompressionLevel.x2, QuantizationConfig.EMPTY },
                { "4x compression", CompressionLevel.x4, QuantizationConfig.EMPTY },
                { "8x compression", CompressionLevel.x8, QuantizationConfig.EMPTY },
                { "16x compression", CompressionLevel.x16, QuantizationConfig.EMPTY },
                { "32x compression", CompressionLevel.x32, QuantizationConfig.EMPTY },
                {
                    "ADC enabled",
                    CompressionLevel.x32,
                    QuantizationConfig.builder().quantizationType(ScalarQuantizationType.ONE_BIT).enableADC(true).build() },
                {
                    "ADC disabled",
                    CompressionLevel.x32,
                    QuantizationConfig.builder().quantizationType(ScalarQuantizationType.ONE_BIT).enableADC(false).build() } }
        );
    }

    @Mock
    private SegmentReader leafReader;
    @Mock
    private MapperService mapperService;
    @Mock
    private Directory directory;
    @Mock
    private FSDirectory fsDirectory;
    @Mock
    private SegmentReader segmentReader;
    @Mock
    private FieldInfos fieldInfos;
    @Mock
    private FieldInfo fieldInfo;
    @Mock
    private KNNVectorFieldType knnVectorFieldType;
    @Mock
    private FloatVectorValues floatVectorValues;
    @Mock
    private KnnVectorValues.DocIndexIterator docIndexIterator;
    @Mock
    private org.opensearch.knn.index.mapper.KNNMappingConfig knnMappingConfig;

    private MemoryOptimizedSearchWarmup warmup;
    private String indexName = "test-index";

    private SegmentCommitInfo createSegmentCommitInfo(String segmentName) {
        SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            segmentName,
            0,
            false,
            false,
            null,
            Collections.emptyMap(),
            new byte[16],
            Collections.emptyMap(),
            null
        );
        segmentInfo.setFiles(Collections.emptySet());
        return new SegmentCommitInfo(segmentInfo, 0, 0, 0, 0, 0, new byte[16]);
    }

    @Before
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        super.setUp();
        warmup = new MemoryOptimizedSearchWarmup();

        ClusterState clusterState = mock(ClusterState.class);
        Metadata metadata = mock(Metadata.class);
        IndexMetadata indexMetadata = mock(IndexMetadata.class);

        Settings indexSettings = org.opensearch.common.settings.Settings.builder().put("index.knn.memory_optimized_search", true).build();

        when(clusterService.state()).thenReturn(clusterState);
        when(clusterState.getMetadata()).thenReturn(metadata);
        when(metadata.index(indexName)).thenReturn(indexMetadata);
        when(indexMetadata.getSettings()).thenReturn(indexSettings);

        // Setup quantization config for the field type
        when(knnMappingConfig.getCompressionLevel()).thenReturn(compressionLevel);
        when(knnVectorFieldType.getKnnMappingConfig()).thenReturn(knnMappingConfig);
    }

    private void setupSingleKnnField(String fieldName) throws IOException {
        when(leafReader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.iterator()).thenReturn(Collections.singletonList(fieldInfo).iterator());
        when(fieldInfo.attributes()).thenReturn(Map.of(KNNVectorFieldMapper.KNN_FIELD, "true", VECTOR_DATA_TYPE_FIELD, "float"));
        when(fieldInfo.getName()).thenReturn(fieldName);
        when(mapperService.fieldType(fieldName)).thenReturn(knnVectorFieldType);
        when(knnVectorFieldType.getIndexCreatedVersion()).thenReturn(org.opensearch.Version.CURRENT);
        when(knnVectorFieldType.isMemoryOptimizedSearchAvailable()).thenReturn(true);

        SegmentCommitInfo segmentCommitInfo = createSegmentCommitInfo("test-segment");
        when(leafReader.getSegmentInfo()).thenReturn(segmentCommitInfo);
    }

    private void setupVectorValuesWithNoDocuments(String fieldName) throws IOException {
        when(leafReader.getFloatVectorValues(fieldName)).thenReturn(floatVectorValues);
        when(floatVectorValues.iterator()).thenReturn(docIndexIterator);
        when(docIndexIterator.nextDoc()).thenReturn(DocIdSetIterator.NO_MORE_DOCS);
    }

    private void setupMultipleKnnFields(String... fieldNames) throws IOException {
        List<FieldInfo> fields = new ArrayList<>();
        for (String fieldName : fieldNames) {
            FieldInfo field = mock(FieldInfo.class);
            when(field.attributes()).thenReturn(Map.of(KNNVectorFieldMapper.KNN_FIELD, "true", VECTOR_DATA_TYPE_FIELD, "float"));
            when(field.getName()).thenReturn(fieldName);
            fields.add(field);
        }
        when(leafReader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.iterator()).thenReturn(fields.iterator());
        when(mapperService.fieldType(anyString())).thenReturn(knnVectorFieldType);
        when(knnVectorFieldType.getIndexCreatedVersion()).thenReturn(org.opensearch.Version.CURRENT);
        when(knnVectorFieldType.isMemoryOptimizedSearchAvailable()).thenReturn(true);

        SegmentCommitInfo segmentCommitInfo = createSegmentCommitInfo("test-segment");
        when(leafReader.getSegmentInfo()).thenReturn(segmentCommitInfo);
    }

    public void testFullPrecisionVectorLoading() throws IOException {
        setupSingleKnnField("test-field");
        when(knnVectorFieldType.isMemoryOptimizedSearchAvailable()).thenReturn(true);
        when(knnVectorFieldType.getIndexCreatedVersion()).thenReturn(org.opensearch.Version.CURRENT);
        when(knnMappingConfig.getQuantizationConfig()).thenReturn(quantizationConfig);

        when(leafReader.getFloatVectorValues("test-field")).thenReturn(floatVectorValues);
        when(floatVectorValues.iterator()).thenReturn(docIndexIterator);
        when(docIndexIterator.nextDoc()).thenReturn(0, 1, DocIdSetIterator.NO_MORE_DOCS);
        when(docIndexIterator.docID()).thenReturn(0, 1);
        when(floatVectorValues.vectorValue(anyInt())).thenReturn(new float[] { 1.0f, 2.0f });

        ArrayList<String> result = warmup.warmUp(leafReader, mapperService, indexName, directory);

        assertNotNull(result);
        verify(floatVectorValues, org.mockito.Mockito.atLeast(2)).vectorValue(anyInt());
    }

    public void testWarmUpWithFSDirectory() throws IOException {
        setupSingleKnnField("test-field");
        setupVectorValuesWithNoDocuments("test-field");

        ArrayList<String> result = warmup.warmUp(leafReader, mapperService, indexName, fsDirectory);

        assertNotNull(result);
        verify(leafReader, atLeastOnce()).getFieldInfos();
    }

    public void testWarmUpWithNonFSDirectory() throws IOException {
        setupSingleKnnField("test-field");
        setupVectorValuesWithNoDocuments("test-field");

        ArrayList<String> result = warmup.warmUp(leafReader, mapperService, indexName, directory);

        assertNotNull(result);
        verify(leafReader, atLeastOnce()).getFieldInfos();
    }

    public void testWarmUpWithMultipleSegments() throws IOException {
        setupMultipleKnnFields("field1", "field2");
        when(leafReader.getFloatVectorValues(anyString())).thenReturn(floatVectorValues);
        when(floatVectorValues.iterator()).thenReturn(docIndexIterator);
        when(docIndexIterator.nextDoc()).thenReturn(DocIdSetIterator.NO_MORE_DOCS);

        ArrayList<String> result = warmup.warmUp(leafReader, mapperService, indexName, directory);

        assertNotNull(result);
        verify(leafReader, atLeastOnce()).getFieldInfos();
    }

    public void testWarmUpWithMultipleFields() throws IOException {
        setupMultipleKnnFields("field1", "field2", "field3");
        when(leafReader.getFloatVectorValues(anyString())).thenReturn(floatVectorValues);
        when(floatVectorValues.iterator()).thenReturn(docIndexIterator);
        when(docIndexIterator.nextDoc()).thenReturn(DocIdSetIterator.NO_MORE_DOCS);

        ArrayList<String> result = warmup.warmUp(leafReader, mapperService, indexName, directory);

        assertNotNull(result);
    }

    public void testWarmUpWithNoEligibleFields() throws IOException {
        when(leafReader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.iterator()).thenReturn(Collections.emptyIterator());

        ArrayList<String> result = warmup.warmUp(leafReader, mapperService, indexName, directory);

        assertNotNull(result);
        assertTrue(result.isEmpty());
    }

    public void testWarmUpWithEmptySegment() throws IOException {
        setupSingleKnnField("test-field");
        when(leafReader.getFloatVectorValues("test-field")).thenReturn(null);

        ArrayList<String> result = warmup.warmUp(leafReader, mapperService, indexName, directory);

        assertNotNull(result);
    }

    public void testWarmUpWithMissingEngineFiles() throws IOException {
        setupSingleKnnField("test-field");
        setupVectorValuesWithNoDocuments("test-field");

        ArrayList<String> result = warmup.warmUp(leafReader, mapperService, indexName, directory);

        assertNotNull(result);
    }

    public void testWarmUpWithInvalidVectorDataType() throws IOException {
        when(leafReader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.iterator()).thenReturn(Collections.singletonList(fieldInfo).iterator());
        when(fieldInfo.attributes()).thenReturn(Map.of(KNNVectorFieldMapper.KNN_FIELD, "true"));
        when(fieldInfo.getName()).thenReturn("test-field");
        when(mapperService.fieldType("test-field")).thenReturn(knnVectorFieldType);
        setupVectorValuesWithNoDocuments("test-field");

        ArrayList<String> result = warmup.warmUp(leafReader, mapperService, indexName, directory);

        assertNotNull(result);
    }

    public void testWarmUpWithClosedLeafReader() throws IOException {
        when(leafReader.getFieldInfos()).thenThrow(new AlreadyClosedException("Reader closed"));

        try {
            ArrayList<String> result = warmup.warmUp(leafReader, mapperService, indexName, directory);
            fail("Expected AlreadyClosedException");
        } catch (AlreadyClosedException e) {
            assertTrue(e.getMessage().contains("Reader closed"));
        }
    }

    public void testIOExceptionInFullWarmUp() throws IOException {
        setupSingleKnnField("test-field");
        setupVectorValuesWithNoDocuments("test-field");

        ArrayList<String> result = warmup.warmUp(leafReader, mapperService, indexName, directory);

        assertNotNull(result);
    }

    public void testIOExceptionInVectorLoading() throws IOException {
        setupSingleKnnField("test-field");
        when(leafReader.getFloatVectorValues("test-field")).thenThrow(new IOException("Vector loading failed"));

        ArrayList<String> result = warmup.warmUp(leafReader, mapperService, indexName, directory);

        assertNotNull(result);
    }

    public void testNullMapperService() throws IOException {
        when(leafReader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.iterator()).thenReturn(Collections.singletonList(fieldInfo).iterator());
        when(fieldInfo.attributes()).thenReturn(Map.of(KNNVectorFieldMapper.KNN_FIELD, "true", VECTOR_DATA_TYPE_FIELD, "float"));
        when(fieldInfo.getName()).thenReturn("test-field");

        ArrayList<String> result = warmup.warmUp(leafReader, null, indexName, directory);

        assertNotNull(result);
        assertTrue(result.isEmpty());
    }

    public void testWarmUpWithLargeSegment() throws IOException {
        setupSingleKnnField("test-field");
        when(leafReader.getFloatVectorValues("test-field")).thenReturn(floatVectorValues);
        when(floatVectorValues.iterator()).thenReturn(docIndexIterator);

        List<Integer> docIds = new ArrayList<>();
        for (int i = 0; i < 10000; i++) {
            docIds.add(i);
        }
        docIds.add(DocIdSetIterator.NO_MORE_DOCS);
        when(docIndexIterator.nextDoc()).thenReturn(
            docIds.toArray(new Integer[0])[0],
            Arrays.copyOfRange(docIds.toArray(new Integer[0]), 1, docIds.size())
        );
        when(floatVectorValues.vectorValue(anyInt())).thenReturn(new float[] { 1.0f, 2.0f });

        long startTime = System.currentTimeMillis();
        ArrayList<String> result = warmup.warmUp(leafReader, mapperService, indexName, directory);
        long duration = System.currentTimeMillis() - startTime;

        assertNotNull(result);
        assertTrue(duration < 30000);
    }

    public void testWarmUpWithManyFields() throws IOException {
        List<FieldInfo> fields = new ArrayList<>();
        for (int i = 0; i < 50; i++) {
            FieldInfo field = mock(FieldInfo.class);
            when(field.attributes()).thenReturn(Map.of(KNNVectorFieldMapper.KNN_FIELD, "true", VECTOR_DATA_TYPE_FIELD, "float"));
            when(field.getName()).thenReturn("field" + i);
            fields.add(field);
        }

        when(leafReader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.iterator()).thenReturn(fields.iterator());
        when(mapperService.fieldType(anyString())).thenReturn(knnVectorFieldType);

        when(leafReader.getFloatVectorValues(anyString())).thenReturn(floatVectorValues);
        when(floatVectorValues.iterator()).thenReturn(docIndexIterator);
        when(docIndexIterator.nextDoc()).thenReturn(DocIdSetIterator.NO_MORE_DOCS);

        long startTime = System.currentTimeMillis();
        ArrayList<String> result = warmup.warmUp(leafReader, mapperService, indexName, directory);
        long duration = System.currentTimeMillis() - startTime;

        assertNotNull(result);
        assertTrue(duration < 10000);
    }

    public void testWarmUpStrategyInvocation() throws IOException {
        setupSingleKnnField("test-field");
        setupVectorValuesWithNoDocuments("test-field");

        ArrayList<String> result = warmup.warmUp(leafReader, mapperService, indexName, directory);

        assertNotNull(result);
        verify(leafReader, atLeastOnce()).getFieldInfos();
    }

    public void testStrategySelectionBasedOnDirectory() throws IOException {
        setupSingleKnnField("test-field");
        setupVectorValuesWithNoDocuments("test-field");

        ArrayList<String> resultFS = warmup.warmUp(leafReader, mapperService, indexName, fsDirectory);
        ArrayList<String> resultNonFS = warmup.warmUp(leafReader, mapperService, indexName, directory);

        assertNotNull(resultFS);
        assertNotNull(resultNonFS);
    }

    public void testWarmUpDuringShardRecovery() throws IOException {
        setupSingleKnnField("test-field");
        setupVectorValuesWithNoDocuments("test-field");

        ArrayList<String> result = warmup.warmUp(leafReader, mapperService, indexName, directory);

        assertNotNull(result);
    }

    public void testWarmUpWithConcurrentQueries() throws IOException {
        setupSingleKnnField("test-field");
        setupVectorValuesWithNoDocuments("test-field");

        ArrayList<String> result = warmup.warmUp(leafReader, mapperService, indexName, directory);

        assertNotNull(result);
    }

    public void testDirectoryResourceCleanup() throws IOException {
        IndexInput mockInput = mock(IndexInput.class);
        when(mockInput.length()).thenReturn(8192L);
        when(directory.openInput(anyString(), any())).thenReturn(mockInput);

        setupSingleKnnField("test-field");
        setupVectorValuesWithNoDocuments("test-field");

        ArrayList<String> result = warmup.warmUp(leafReader, mapperService, indexName, directory);

        assertNotNull(result);
    }

    public void testSearcherResourceCleanup() throws IOException {
        setupSingleKnnField("test-field");
        setupVectorValuesWithNoDocuments("test-field");

        ArrayList<String> result = warmup.warmUp(leafReader, mapperService, indexName, directory);

        assertNotNull(result);
        verify(leafReader, atLeastOnce()).getFieldInfos();
    }
}

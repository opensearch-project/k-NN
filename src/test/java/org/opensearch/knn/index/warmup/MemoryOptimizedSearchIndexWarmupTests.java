/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.warmup;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
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
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.engine.qframe.QuantizationConfigParser;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static com.carrotsearch.randomizedtesting.RandomizedTest.$;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.QFRAMEWORK_CONFIG;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

/**
 * Parameterized tests for MemoryOptimizedSearchWarmup to ensure warmup doesn't crash
 * with different index configurations including quantized indices, native byte/binary types,
 * and regular float vectors.
 */
public class MemoryOptimizedSearchIndexWarmupTests extends KNNTestCase {

    @Mock
    private SegmentReader leafReader;
    @Mock
    private MapperService mapperService;
    @Mock
    private Directory directory;
    @Mock
    private FieldInfos fieldInfos;
    @Mock
    private FieldInfo fieldInfo;
    @Mock
    private KNNVectorFieldType knnVectorFieldType;
    @Mock
    private FloatVectorValues floatVectorValues;
    @Mock
    private ByteVectorValues byteVectorValues;
    @Mock
    private KnnVectorValues.DocIndexIterator docIndexIterator;

    private MemoryOptimizedSearchWarmup warmup;
    private String indexName = "test-index";
    private static final String FIELD_NAME = "test_vector_field";

    // Test parameters
    private final String description;
    private final VectorDataType vectorDataType;
    private final QuantizationConfig quantizationConfig;
    private final boolean useByteVectors;

    // Constructor for parameterized tests
    public MemoryOptimizedSearchIndexWarmupTests(
        String description,
        VectorDataType vectorDataType,
        QuantizationConfig quantizationConfig,
        boolean useByteVectors
    ) {
        this.description = description;
        this.vectorDataType = vectorDataType;
        this.quantizationConfig = quantizationConfig;
        this.useByteVectors = useByteVectors;
    }

    @Before
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        super.setUp();
        warmup = new MemoryOptimizedSearchWarmup();

        ClusterState clusterState = mock(ClusterState.class);
        Metadata metadata = mock(Metadata.class);
        IndexMetadata indexMetadata = mock(IndexMetadata.class);

        Settings indexSettings = Settings.builder().put("index.knn.memory_optimized_search", true).build();

        when(clusterService.state()).thenReturn(clusterState);
        when(clusterState.getMetadata()).thenReturn(metadata);
        when(metadata.index(indexName)).thenReturn(indexMetadata);
        when(indexMetadata.getSettings()).thenReturn(indexSettings);
    }

    /**
     * Factory method to generate test parameters for various index configurations.
     */
    @ParametersFactory(argumentFormatting = "description:%1$s")
    public static Iterable<Object[]> parameters() {
        return Arrays.asList(
            $(
                "1-bit quantized index (FLOAT vector type)",
                VectorDataType.FLOAT,
                createQuantizationConfig(ScalarQuantizationType.ONE_BIT, false, false),
                true  // Uses byte vectors
            ),
            $(
                "2-bit quantized index (FLOAT vector type)",
                VectorDataType.FLOAT,
                createQuantizationConfig(ScalarQuantizationType.TWO_BIT, false, false),
                true  // Uses byte vectors
            ),
            $(
                "4-bit quantized index (FLOAT vector type)",
                VectorDataType.FLOAT,
                createQuantizationConfig(ScalarQuantizationType.FOUR_BIT, false, false),
                true  // Uses byte vectors
            ),
            $(
                "1-bit quantized index with ADC (FLOAT vector type)",
                VectorDataType.FLOAT,
                createQuantizationConfig(ScalarQuantizationType.ONE_BIT, true, false),
                true  // Uses byte vectors
            ),
            $(
                "1-bit quantized index with random rotation (FLOAT vector type)",
                VectorDataType.FLOAT,
                createQuantizationConfig(ScalarQuantizationType.ONE_BIT, false, true),
                true  // Uses byte vectors
            ),
            $(
                "Native BYTE vector type (no quantization)",
                VectorDataType.BYTE,
                QuantizationConfig.EMPTY,
                true  // Uses byte vectors
            ),
            $(
                "Native BINARY vector type (no quantization)",
                VectorDataType.BINARY,
                QuantizationConfig.EMPTY,
                true  // Uses byte vectors
            ),
            $(
                "Regular FLOAT vectors (no quantization)",
                VectorDataType.FLOAT,
                QuantizationConfig.EMPTY,
                false  // Uses float vectors
            ),
            $(
                "ADC transformed FLOAT vectors (no quantization config)",
                VectorDataType.FLOAT,
                null,
                false  // Uses float vectors
            )
        );
    }

    /**
     * Helper method to create QuantizationConfig for testing.
     */
    private static QuantizationConfig createQuantizationConfig(
        ScalarQuantizationType sqType,
        boolean enableADC,
        boolean enableRandomRotation
    ) {
        return QuantizationConfig.builder()
            .quantizationType(sqType)
            .enableADC(enableADC)
            .enableRandomRotation(enableRandomRotation)
            .build();
    }

    /**
     * Parameterized test that verifies warmup doesn't crash with different index configurations.
     * Tests that the warmup successfully completes without throwing exceptions for all
     * combinations of vector types and quantization configurations.
     */
    public void testWarmupWithDifferentIndexConfigurations() throws IOException {
        // Setup field info with the test configuration
        setupFieldInfo();

        // Setup vector values based on whether we're using byte or float vectors
        setupVectorValues();

        // Execute warmup - should not throw any exceptions
        ArrayList<String> result = warmup.warmUp(leafReader, mapperService, indexName, directory);

        // Verify warmup completed successfully
        assertNotNull("Warmup result should not be null for: " + description, result);

        // The warmup should complete without crashing regardless of index configuration
        // This ensures the SearchVectorTypeResolver correctly handles all scenarios
    }

    /**
     * Test warmup with actual vector data to ensure it processes documents correctly.
     */
    public void testWarmupWithVectorData() throws IOException {
        setupFieldInfo();

        if (useByteVectors) {
            // Setup byte vector values with some documents
            when(leafReader.getByteVectorValues(FIELD_NAME)).thenReturn(byteVectorValues);
            when(byteVectorValues.iterator()).thenReturn(docIndexIterator);
            when(docIndexIterator.nextDoc()).thenReturn(0, 1, DocIdSetIterator.NO_MORE_DOCS);
            when(docIndexIterator.docID()).thenReturn(0, 1);
            when(byteVectorValues.vectorValue(0)).thenReturn(new byte[] { 1, 2, 3, 4 });
            when(byteVectorValues.vectorValue(1)).thenReturn(new byte[] { 5, 6, 7, 8 });
        } else {
            // Setup float vector values with some documents
            when(leafReader.getFloatVectorValues(FIELD_NAME)).thenReturn(floatVectorValues);
            when(floatVectorValues.iterator()).thenReturn(docIndexIterator);
            when(docIndexIterator.nextDoc()).thenReturn(0, 1, DocIdSetIterator.NO_MORE_DOCS);
            when(docIndexIterator.docID()).thenReturn(0, 1);
            when(floatVectorValues.vectorValue(0)).thenReturn(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
            when(floatVectorValues.vectorValue(1)).thenReturn(new float[] { 5.0f, 6.0f, 7.0f, 8.0f });
        }

        // Execute warmup
        ArrayList<String> result = warmup.warmUp(leafReader, mapperService, indexName, directory);

        // Verify warmup completed successfully
        assertNotNull("Warmup result should not be null for: " + description, result);
    }

    /**
     * Test warmup with empty index (no documents).
     */
    public void testWarmupWithEmptyIndex() throws IOException {
        setupFieldInfo();

        if (useByteVectors) {
            when(leafReader.getByteVectorValues(FIELD_NAME)).thenReturn(byteVectorValues);
            when(byteVectorValues.iterator()).thenReturn(docIndexIterator);
            when(docIndexIterator.nextDoc()).thenReturn(DocIdSetIterator.NO_MORE_DOCS);
        } else {
            when(leafReader.getFloatVectorValues(FIELD_NAME)).thenReturn(floatVectorValues);
            when(floatVectorValues.iterator()).thenReturn(docIndexIterator);
            when(docIndexIterator.nextDoc()).thenReturn(DocIdSetIterator.NO_MORE_DOCS);
        }

        // Execute warmup
        ArrayList<String> result = warmup.warmUp(leafReader, mapperService, indexName, directory);

        // Verify warmup completed successfully even with no documents
        assertNotNull("Warmup result should not be null for empty index: " + description, result);
    }

    /**
     * Helper method to setup FieldInfo mock with the test configuration.
     */
    private void setupFieldInfo() throws IOException {
        Map<String, String> attributes = new HashMap<>();
        attributes.put(KNNVectorFieldMapper.KNN_FIELD, "true");
        attributes.put(VECTOR_DATA_TYPE_FIELD, vectorDataType.getValue());

        // Add quantization config if present
        if (quantizationConfig != null && quantizationConfig != QuantizationConfig.EMPTY) {
            String configCsv = QuantizationConfigParser.toCsv(quantizationConfig);
            attributes.put(QFRAMEWORK_CONFIG, configCsv);
        }

        when(fieldInfo.attributes()).thenReturn(attributes);
        when(fieldInfo.getName()).thenReturn(FIELD_NAME);

        when(leafReader.getFieldInfos()).thenReturn(fieldInfos);
        when(fieldInfos.iterator()).thenReturn(Collections.singletonList(fieldInfo).iterator());

        when(mapperService.fieldType(FIELD_NAME)).thenReturn(knnVectorFieldType);
        when(knnVectorFieldType.getIndexCreatedVersion()).thenReturn(org.opensearch.Version.CURRENT);
        when(knnVectorFieldType.isMemoryOptimizedSearchAvailable()).thenReturn(true);

        SegmentCommitInfo segmentCommitInfo = createSegmentCommitInfo("test-segment");
        when(leafReader.getSegmentInfo()).thenReturn(segmentCommitInfo);
    }

    /**
     * Helper method to setup vector values based on the test configuration.
     */
    private void setupVectorValues() throws IOException {
        if (useByteVectors) {
            when(leafReader.getByteVectorValues(FIELD_NAME)).thenReturn(byteVectorValues);
            when(byteVectorValues.iterator()).thenReturn(docIndexIterator);
            when(docIndexIterator.nextDoc()).thenReturn(DocIdSetIterator.NO_MORE_DOCS);
        } else {
            when(leafReader.getFloatVectorValues(FIELD_NAME)).thenReturn(floatVectorValues);
            when(floatVectorValues.iterator()).thenReturn(docIndexIterator);
            when(docIndexIterator.nextDoc()).thenReturn(DocIdSetIterator.NO_MORE_DOCS);
        }
    }

    /**
     * Helper method to create SegmentCommitInfo for testing.
     */
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
}

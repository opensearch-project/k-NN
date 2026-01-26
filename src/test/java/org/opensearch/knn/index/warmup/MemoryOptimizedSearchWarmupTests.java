/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.warmup;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.tests.store.BaseDirectoryWrapper;
import org.mockito.MockedStatic;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;
import org.opensearch.knn.index.codec.util.UnitTestCodec;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.MemoryOptimizedSearchSupportSpec;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.engine.qframe.QuantizationConfigParser;
import org.opensearch.knn.index.mapper.KNNMappingConfig;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.when;

/**
 * Parameterized tests for {@link MemoryOptimizedSearchWarmup} using actual Lucene segment readers.
 * Tests various index configurations including different compression levels, binary vectors,
 * disk-based indices, and ADC (Asymmetric Distance Computation) configurations.
 */
@RequiredArgsConstructor
public class MemoryOptimizedSearchWarmupTests extends KNNTestCase {

    private static final Codec TESTING_CODEC = new UnitTestCodec(() -> new NativeEngines990KnnVectorsFormat(0));
    private static final String TEST_INDEX = "test-index";
    private static final String KNN_FIELD = "knn_field";

    // Test parameters
    private final String description;
    private final IndexConfig indexConfig;

    private MemoryOptimizedSearchWarmup warmup;
    private Directory directory;
    private RandomIndexWriter indexWriter;

    @ParametersFactory(argumentFormatting = "%1$s")
    public static Collection<Object[]> parameters() {
        return Arrays.asList(
            // 1x compression (no compression) - Float HNSW with Flat encoder
            new Object[] { "1x_compression_float", new IndexConfig(VectorDataType.FLOAT, "HNSW16,Flat", null) },

            // 2x compression - Float HNSW with SQ (Scalar Quantization) FP16
            new Object[] { "2x_compression_fp16", new IndexConfig(VectorDataType.FLOAT, "HNSW16,SQfp16", null) },

            // 4x compression - Byte vectors (SQ8)
            new Object[] { "4x_compression_byte", new IndexConfig(VectorDataType.BYTE, "HNSW16,SQ8_direct_signed", null) },

            // 8x compression - Binary quantization with 4 bits
            new Object[] {
                "8x_compression_binary_4bit",
                new IndexConfig(
                    VectorDataType.FLOAT,
                    "BHNSW16,Flat",
                    QuantizationConfig.builder().quantizationType(ScalarQuantizationType.FOUR_BIT).build()
                ) },

            // 16x compression - Binary quantization with 2 bits
            new Object[] {
                "16x_compression_binary_2bit",
                new IndexConfig(
                    VectorDataType.FLOAT,
                    "BHNSW16,Flat",
                    QuantizationConfig.builder().quantizationType(ScalarQuantizationType.TWO_BIT).build()
                ) },

            // 32x compression - Binary quantization with 1 bit
            new Object[] {
                "32x_compression_binary_1bit",
                new IndexConfig(
                    VectorDataType.FLOAT,
                    "BHNSW16,Flat",
                    QuantizationConfig.builder().quantizationType(ScalarQuantizationType.ONE_BIT).build()
                ) },

            // Disk-based 32x compression
            new Object[] {
                "disk_based_32x",
                new IndexConfig(
                    VectorDataType.FLOAT,
                    "BHNSW16,Flat",
                    QuantizationConfig.builder().quantizationType(ScalarQuantizationType.ONE_BIT).build(),
                    Mode.ON_DISK
                ) }

        );
    }

    @Override
    public void setUp() throws Exception {
        super.setUp();
        warmup = new MemoryOptimizedSearchWarmup();
    }

    @Override
    public void tearDown() throws Exception {
        if (directory != null) {
            directory.close();
        }
        super.tearDown();
    }

    public void testWarmUp_whenMapperServiceIsNull_thenReturnEmptyList() throws IOException {
        setupDirectory();
        addVectorDocument(KNN_FIELD);

        try (IndexReader reader = indexWriter.getReader()) {
            indexWriter.close();
            LeafReader leafReader = reader.leaves().get(0).reader();

            ArrayList<String> result = warmup.warmUp(leafReader, null, TEST_INDEX, directory);
            assertTrue("Expected empty result when mapper service is null for " + description, result.isEmpty());
        }
    }

    public void testWarmUp_whenFieldTypeIsNotKnnVectorFieldType_thenSkipField() throws IOException {
        setupDirectory();
        addVectorDocument(KNN_FIELD);

        try (IndexReader reader = indexWriter.getReader()) {
            indexWriter.close();
            LeafReader leafReader = reader.leaves().get(0).reader();
            MapperService mapperService = mock(MapperService.class);

            // Return a non-KNN field type
            MappedFieldType nonKnnFieldType = mock(MappedFieldType.class);
            when(mapperService.fieldType(KNN_FIELD)).thenReturn(nonKnnFieldType);

            ArrayList<String> result = warmup.warmUp(leafReader, mapperService, TEST_INDEX, directory);
            assertTrue("Expected empty result when field type is not KNN for " + description, result.isEmpty());
        }
    }

    public void testWarmUp_whenMemoryOptimizedSearchNotSupported_thenSkipField() throws IOException {
        setupDirectory();
        addVectorDocument(KNN_FIELD);

        try (IndexReader reader = indexWriter.getReader()) {
            indexWriter.close();
            LeafReader leafReader = reader.leaves().get(0).reader();
            MapperService mapperService = mock(MapperService.class);

            KNNVectorFieldType knnFieldType = createMockedKnnFieldType();
            when(mapperService.fieldType(KNN_FIELD)).thenReturn(knnFieldType);

            try (MockedStatic<MemoryOptimizedSearchSupportSpec> supportSpecMock = mockStatic(MemoryOptimizedSearchSupportSpec.class)) {
                supportSpecMock.when(() -> MemoryOptimizedSearchSupportSpec.isSupportedFieldType(eq(knnFieldType), anyString()))
                    .thenReturn(false);

                ArrayList<String> result = warmup.warmUp(leafReader, mapperService, TEST_INDEX, directory);
                assertTrue("Expected empty result when memory optimized search not supported for " + description, result.isEmpty());
            }
        }
    }

    @SneakyThrows
    public void testWarmUp_thenWarmUpSuccessfully() {
        setupDirectory();
        addVectorDocument(KNN_FIELD);

        try (IndexReader reader = indexWriter.getReader()) {
            indexWriter.flush();
            indexWriter.commit();
            indexWriter.close();

            IndexSearcher searcher = new IndexSearcher(reader);
            LeafReader leafReader = searcher.getLeafContexts().get(0).reader();
            SegmentReader segmentReader = Lucene.segmentReader(leafReader);

            // Verify we have a real segment reader
            assertNotNull("SegmentReader should not be null for " + description, segmentReader);
            assertNotNull("SegmentInfo should not be null for " + description, segmentReader.getSegmentInfo());

            MapperService mapperService = mock(MapperService.class);
            KNNVectorFieldType knnFieldType = createMockedKnnFieldType();
            when(mapperService.fieldType(KNN_FIELD)).thenReturn(knnFieldType);

            try (MockedStatic<MemoryOptimizedSearchSupportSpec> supportSpecMock = mockStatic(MemoryOptimizedSearchSupportSpec.class)) {
                supportSpecMock.when(() -> MemoryOptimizedSearchSupportSpec.isSupportedFieldType(eq(knnFieldType), anyString()))
                    .thenReturn(true);

                ArrayList<String> result = warmup.warmUp(leafReader, mapperService, TEST_INDEX, directory);
                assertEquals("Expected 1 warmed up field for " + description, 1, result.size());
                assertEquals("Expected field name to match for " + description, KNN_FIELD, result.get(0));
            }
        }
    }

    @SneakyThrows
    public void testWarmUp_whenMultipleVectors_thenWarmUpSuccessfully() {
        setupDirectory();

        // Add multiple vectors to the same field
        addVectorDocument(KNN_FIELD);
        addVectorDocument(KNN_FIELD);
        addVectorDocument(KNN_FIELD);

        try (IndexReader reader = indexWriter.getReader()) {
            indexWriter.flush();
            indexWriter.commit();
            indexWriter.close();

            IndexSearcher searcher = new IndexSearcher(reader);
            LeafReader leafReader = searcher.getLeafContexts().get(0).reader();

            MapperService mapperService = mock(MapperService.class);
            KNNVectorFieldType knnFieldType = createMockedKnnFieldType();
            when(mapperService.fieldType(KNN_FIELD)).thenReturn(knnFieldType);

            try (MockedStatic<MemoryOptimizedSearchSupportSpec> supportSpecMock = mockStatic(MemoryOptimizedSearchSupportSpec.class)) {
                supportSpecMock.when(() -> MemoryOptimizedSearchSupportSpec.isSupportedFieldType(eq(knnFieldType), anyString()))
                    .thenReturn(true);

                ArrayList<String> result = warmup.warmUp(leafReader, mapperService, TEST_INDEX, directory);
                assertEquals("Expected 1 warmed up field for " + description, 1, result.size());
                assertEquals("Expected field name to match for " + description, KNN_FIELD, result.get(0));
            }
        }
    }

    @SneakyThrows
    public void testWarmUp_whenMultipleFields_thenWarmUpAllSupportedFields() {
        setupDirectory();

        String field1 = "knn_field_1";
        String field2 = "knn_field_2";

        // Add a document with both fields to ensure proper vector file structure
        addVectorDocumentWithMultipleFields(field1, field2);

        try (IndexReader reader = indexWriter.getReader()) {
            indexWriter.flush();
            indexWriter.commit();
            indexWriter.close();

            IndexSearcher searcher = new IndexSearcher(reader);
            LeafReader leafReader = searcher.getLeafContexts().get(0).reader();

            MapperService mapperService = mock(MapperService.class);
            KNNVectorFieldType knnFieldType1 = createMockedKnnFieldType();
            KNNVectorFieldType knnFieldType2 = createMockedKnnFieldType();
            when(mapperService.fieldType(field1)).thenReturn(knnFieldType1);
            when(mapperService.fieldType(field2)).thenReturn(knnFieldType2);

            try (MockedStatic<MemoryOptimizedSearchSupportSpec> supportSpecMock = mockStatic(MemoryOptimizedSearchSupportSpec.class)) {
                supportSpecMock.when(() -> MemoryOptimizedSearchSupportSpec.isSupportedFieldType(eq(knnFieldType1), anyString()))
                    .thenReturn(true);
                supportSpecMock.when(() -> MemoryOptimizedSearchSupportSpec.isSupportedFieldType(eq(knnFieldType2), anyString()))
                    .thenReturn(true);

                ArrayList<String> result = warmup.warmUp(leafReader, mapperService, TEST_INDEX, directory);
                assertEquals("Expected 2 warmed up fields for " + description, 2, result.size());
                assertTrue("Expected field1 to be warmed up for " + description, result.contains(field1));
                assertTrue("Expected field2 to be warmed up for " + description, result.contains(field2));
            }
        }
    }

    // ==================== Helper Methods ====================

    private void setupDirectory() throws IOException {
        directory = newFSDirectory(createTempDir());
        // Disable index checking on close since we're using native engine format
        ((BaseDirectoryWrapper) directory).setCheckIndexOnClose(false);
        indexWriter = createIndexWriter(directory);
    }

    private RandomIndexWriter createIndexWriter(Directory dir) throws IOException {
        IndexWriterConfig iwc = newIndexWriterConfig();
        iwc.setMergeScheduler(new SerialMergeScheduler());
        iwc.setCodec(TESTING_CODEC);
        iwc.setUseCompoundFile(false);
        // Set merge policy to no merges so that we create a predictable number of segments
        iwc.setMergePolicy(NoMergePolicy.INSTANCE);
        return new RandomIndexWriter(random(), dir, iwc);
    }

    /**
     * Creates a mocked KNNVectorFieldType configured with the current test's mode.
     * This ensures disk-based mode is properly reflected in the field type for future-proofing.
     */
    private KNNVectorFieldType createMockedKnnFieldType() {
        KNNVectorFieldType knnFieldType = mock(KNNVectorFieldType.class);
        KNNMappingConfig mappingConfig = mock(KNNMappingConfig.class);
        when(mappingConfig.getMode()).thenReturn(indexConfig.mode);
        when(knnFieldType.getKnnMappingConfig()).thenReturn(mappingConfig);
        return knnFieldType;
    }

    private void addVectorDocument(String fieldName) throws IOException {
        Document doc = new Document();
        Field vectorField = createVectorField(fieldName);
        doc.add(vectorField);
        indexWriter.addDocument(doc);
    }

    private void addVectorDocumentWithMultipleFields(String... fieldNames) throws IOException {
        Document doc = new Document();
        for (String fieldName : fieldNames) {
            Field vectorField = createVectorField(fieldName);
            doc.add(vectorField);
        }
        indexWriter.addDocument(doc);
    }

    private Field createVectorField(String fieldName) {
        int dimension = getDimensionForConfig();
        FieldType fieldType = createVectorFieldType(dimension);

        if (indexConfig.vectorDataType == VectorDataType.BYTE) {
            byte[] vector = generateByteVector(dimension);
            return new KnnByteVectorField(fieldName, vector, fieldType);
        } else {
            float[] vector = generateFloatVector(dimension);
            return new KnnFloatVectorField(fieldName, vector, fieldType);
        }
    }

    private int getDimensionForConfig() {
        // Binary quantization requires dimensions divisible by 8
        if (indexConfig.quantizationConfig != null) {
            return 128; // Use 128 dimensions for binary quantization
        }
        return 3; // Default dimension for simple tests
    }

    private float[] generateFloatVector(int dimension) {
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = random().nextFloat() * 10 - 5; // Random values between -5 and 5
        }
        return vector;
    }

    private byte[] generateByteVector(int dimension) {
        byte[] vector = new byte[dimension];
        random().nextBytes(vector);
        return vector;
    }

    private FieldType createVectorFieldType(int dimension) {
        FieldType fieldType = new FieldType();
        fieldType.setTokenized(false);
        fieldType.setIndexOptions(IndexOptions.NONE);
        fieldType.putAttribute(KNNVectorFieldMapper.KNN_FIELD, "true");
        fieldType.putAttribute(KNNConstants.KNN_METHOD, KNNConstants.METHOD_HNSW);
        fieldType.putAttribute(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName());
        fieldType.putAttribute(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue());
        fieldType.putAttribute(KNNConstants.HNSW_ALGO_M, "16");
        fieldType.putAttribute(KNNConstants.HNSW_ALGO_EF_CONSTRUCTION, "100");
        fieldType.putAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD, indexConfig.vectorDataType.getValue());
        fieldType.putAttribute(
            KNNConstants.PARAMETERS,
            "{ \"index_description\":\"" + indexConfig.indexDescription + "\", \"spaceType\": \"l2\"}"
        );

        // Add quantization config if present
        if (indexConfig.quantizationConfig != null) {
            fieldType.putAttribute(KNNConstants.QFRAMEWORK_CONFIG, QuantizationConfigParser.toCsv(indexConfig.quantizationConfig));
        }

        VectorEncoding encoding = (indexConfig.vectorDataType == VectorDataType.BYTE) ? VectorEncoding.BYTE : VectorEncoding.FLOAT32;
        fieldType.setVectorAttributes(dimension, encoding, SpaceType.L2.getKnnVectorSimilarityFunction().getVectorSimilarityFunction());
        fieldType.freeze();
        return fieldType;
    }

    // ==================== Index Configuration ====================

    /**
     * Configuration for different index types being tested.
     */
    private static class IndexConfig {
        final VectorDataType vectorDataType;
        final String indexDescription;
        final QuantizationConfig quantizationConfig;
        final Mode mode;

        IndexConfig(VectorDataType vectorDataType, String indexDescription, QuantizationConfig quantizationConfig) {
            this(vectorDataType, indexDescription, quantizationConfig, Mode.NOT_CONFIGURED);
        }

        IndexConfig(VectorDataType vectorDataType, String indexDescription, QuantizationConfig quantizationConfig, Mode mode) {
            this.vectorDataType = vectorDataType;
            this.indexDescription = indexDescription;
            this.quantizationConfig = quantizationConfig;
            this.mode = mode;
        }
    }
}

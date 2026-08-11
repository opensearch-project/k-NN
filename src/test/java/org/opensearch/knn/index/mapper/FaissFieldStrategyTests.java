/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DocValuesType;
import org.opensearch.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.ResolvedIndexSpec;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

import java.util.Collections;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.QFRAMEWORK_CONFIG;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.SQ_CONFIG;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class FaissFieldStrategyTests extends KNNTestCase {

    private static final int TEST_DIMENSION = 128;

    // A spec whose isSQOneBit() is false — the field strategy then falls through to the qframe_config path
    private static ResolvedIndexSpec nonSQOneBitSpec() {
        return ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .encoderType(Encoder.EncoderType.FLAT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(TEST_DIMENSION)
            .indexVersionCreated(Version.CURRENT)
            .build();
    }

    public void testBuildFieldTypeConfigForFaissWithSQ1Bit() {
        KNNMappingConfig mappingConfig = mock(KNNMappingConfig.class);
        when(mappingConfig.getDimension()).thenReturn(TEST_DIMENSION);

        KNNMethodContext methodContext = mock(KNNMethodContext.class);
        when(methodContext.getSpaceType()).thenReturn(SpaceType.L2);
        when(methodContext.getKnnEngine()).thenReturn(KNNEngine.FAISS);

        ResolvedIndexSpec resolvedSpec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName("hnsw")
            .encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.ONE)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(TEST_DIMENSION)
            .indexVersionCreated(Version.CURRENT)
            .build();

        KNNLibraryIndexingContext libraryContext = mock(KNNLibraryIndexingContext.class);
        when(libraryContext.getQuantizationConfig()).thenReturn(QuantizationConfig.EMPTY);
        when(libraryContext.getLibraryParameters()).thenReturn(Collections.emptyMap());
        when(libraryContext.getResolvedSpec()).thenReturn(resolvedSpec);
        when(libraryContext.getVectorTransformer()).thenReturn(null);

        FieldTypeConfig config = FaissFieldStrategy.INSTANCE.buildFieldTypeConfig(
            mappingConfig,
            methodContext,
            libraryContext,
            VectorDataType.FLOAT,
            Version.CURRENT,
            true
        );

        assertNotNull(config.getFieldType());
        assertNull(config.getVectorFieldType());
        assertTrue(config.isUseLuceneBasedVectorField());

        FieldType fieldType = config.getFieldType();
        assertEquals(String.valueOf(TEST_DIMENSION), fieldType.getAttributes().get(DIMENSION));
        assertEquals(SpaceType.L2.getValue(), fieldType.getAttributes().get(SPACE_TYPE));
        assertEquals(KNNEngine.FAISS.getName(), fieldType.getAttributes().get(KNN_ENGINE));
        assertEquals(VectorDataType.FLOAT.getValue(), fieldType.getAttributes().get(VECTOR_DATA_TYPE_FIELD));
        assertNotNull(fieldType.getAttributes().get(SQ_CONFIG));
    }

    public void testBuildFieldTypeConfigForFaissWithoutQuantization() {
        KNNMappingConfig mappingConfig = mock(KNNMappingConfig.class);
        when(mappingConfig.getDimension()).thenReturn(TEST_DIMENSION);

        KNNMethodContext methodContext = mock(KNNMethodContext.class);
        when(methodContext.getSpaceType()).thenReturn(SpaceType.INNER_PRODUCT);
        when(methodContext.getKnnEngine()).thenReturn(KNNEngine.FAISS);

        ResolvedIndexSpec resolvedSpec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName("hnsw")
            .encoderType(Encoder.EncoderType.FLAT)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(TEST_DIMENSION)
            .indexVersionCreated(Version.CURRENT)
            .build();

        KNNLibraryIndexingContext libraryContext = mock(KNNLibraryIndexingContext.class);
        when(libraryContext.getQuantizationConfig()).thenReturn(QuantizationConfig.EMPTY);
        when(libraryContext.getLibraryParameters()).thenReturn(Collections.emptyMap());
        when(libraryContext.getResolvedSpec()).thenReturn(resolvedSpec);
        when(libraryContext.getVectorTransformer()).thenReturn(null);

        FieldTypeConfig config = FaissFieldStrategy.INSTANCE.buildFieldTypeConfig(
            mappingConfig,
            methodContext,
            libraryContext,
            VectorDataType.FLOAT,
            Version.CURRENT,
            true
        );

        assertNotNull(config.getFieldType());
        assertNull(config.getVectorFieldType());
        assertTrue(config.isUseLuceneBasedVectorField());

        FieldType fieldType = config.getFieldType();
        assertEquals(String.valueOf(TEST_DIMENSION), fieldType.getAttributes().get(DIMENSION));
        assertEquals(SpaceType.INNER_PRODUCT.getValue(), fieldType.getAttributes().get(SPACE_TYPE));
        assertNull(fieldType.getAttributes().get(SQ_CONFIG));
    }

    public void testFaissFieldStrategyDoesNotProvideFieldOverrides() {
        EngineFieldStrategy strategy = FaissFieldStrategy.INSTANCE;
        assertTrue(
            strategy.getFloatFieldOverrides("test_field", new float[] { 1.0f, 2.0f, 3.0f }, null, null, false, false, false).isEmpty()
        );
        assertTrue(strategy.getByteFieldOverrides("test_field", new byte[] { 1, 2, 3 }, null, null, false, false, false).isEmpty());
    }

    public void testBuildFieldTypeConfigForFaissWithBQEncoder() {
        KNNMappingConfig mappingConfig = mock(KNNMappingConfig.class);
        when(mappingConfig.getDimension()).thenReturn(TEST_DIMENSION);

        KNNMethodContext methodContext = mock(KNNMethodContext.class);
        when(methodContext.getSpaceType()).thenReturn(SpaceType.L2);
        when(methodContext.getKnnEngine()).thenReturn(KNNEngine.FAISS);

        ResolvedIndexSpec resolvedSpec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName("hnsw")
            .encoderType(Encoder.EncoderType.BQ)
            .quantizationBits(Encoder.QuantizationBits.ONE)
            .compressionLevel(CompressionLevel.x32)
            .mode(Mode.ON_DISK)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(TEST_DIMENSION)
            .indexVersionCreated(Version.CURRENT)
            .build();

        KNNLibraryIndexingContext libraryContext = mock(KNNLibraryIndexingContext.class);
        when(libraryContext.getQuantizationConfig()).thenReturn(QuantizationConfig.EMPTY);
        when(libraryContext.getLibraryParameters()).thenReturn(Collections.emptyMap());
        when(libraryContext.getResolvedSpec()).thenReturn(resolvedSpec);
        when(libraryContext.getVectorTransformer()).thenReturn(null);

        FieldTypeConfig config = FaissFieldStrategy.INSTANCE.buildFieldTypeConfig(
            mappingConfig,
            methodContext,
            libraryContext,
            VectorDataType.FLOAT,
            Version.CURRENT,
            true
        );

        assertNotNull(config.getFieldType());
        assertTrue(config.isUseLuceneBasedVectorField());
        FieldType fieldType = config.getFieldType();
        assertEquals(String.valueOf(TEST_DIMENSION), fieldType.getAttributes().get(DIMENSION));
        assertNull(fieldType.getAttributes().get(SQ_CONFIG));
    }

    public void testBuildFieldTypeConfigForFaissWithQuantizationConfig() {
        KNNMappingConfig mappingConfig = mock(KNNMappingConfig.class);
        when(mappingConfig.getDimension()).thenReturn(TEST_DIMENSION);

        KNNMethodContext methodContext = mock(KNNMethodContext.class);
        when(methodContext.getSpaceType()).thenReturn(SpaceType.L2);
        when(methodContext.getKnnEngine()).thenReturn(KNNEngine.FAISS);

        QuantizationConfig quantizationConfig = QuantizationConfig.builder().quantizationType(ScalarQuantizationType.ONE_BIT).build();
        KNNLibraryIndexingContext libraryContext = mock(KNNLibraryIndexingContext.class);
        when(libraryContext.getQuantizationConfig()).thenReturn(quantizationConfig);
        when(libraryContext.getLibraryParameters()).thenReturn(Collections.emptyMap());
        when(libraryContext.getResolvedSpec()).thenReturn(nonSQOneBitSpec());
        when(libraryContext.getVectorTransformer()).thenReturn(null);

        FieldTypeConfig config = FaissFieldStrategy.INSTANCE.buildFieldTypeConfig(
            mappingConfig,
            methodContext,
            libraryContext,
            VectorDataType.FLOAT,
            Version.CURRENT,
            true
        );

        FieldType fieldType = config.getFieldType();
        assertNotNull(fieldType.getAttributes().get(QFRAMEWORK_CONFIG));
        assertNull(fieldType.getAttributes().get(SQ_CONFIG));
    }

    public void testBuildFieldTypeConfigForFaissWithPre217Version_thenDocValuesBased() {
        KNNMappingConfig mappingConfig = mock(KNNMappingConfig.class);
        when(mappingConfig.getDimension()).thenReturn(TEST_DIMENSION);

        KNNMethodContext methodContext = mock(KNNMethodContext.class);
        when(methodContext.getSpaceType()).thenReturn(SpaceType.L2);
        when(methodContext.getKnnEngine()).thenReturn(KNNEngine.FAISS);

        KNNLibraryIndexingContext libraryContext = mock(KNNLibraryIndexingContext.class);
        when(libraryContext.getQuantizationConfig()).thenReturn(QuantizationConfig.EMPTY);
        when(libraryContext.getLibraryParameters()).thenReturn(Collections.emptyMap());
        when(libraryContext.getResolvedSpec()).thenReturn(nonSQOneBitSpec());
        when(libraryContext.getVectorTransformer()).thenReturn(null);

        FieldTypeConfig config = FaissFieldStrategy.INSTANCE.buildFieldTypeConfig(
            mappingConfig,
            methodContext,
            libraryContext,
            VectorDataType.FLOAT,
            Version.V_2_16_0,
            true
        );

        assertFalse(config.isUseLuceneBasedVectorField());
        assertEquals(DocValuesType.BINARY, config.getFieldType().docValuesType());
    }

    public void testBuildFieldTypeConfigForFaissWithBinaryDataType_thenDimensionAdjusted() {
        KNNMappingConfig mappingConfig = mock(KNNMappingConfig.class);
        when(mappingConfig.getDimension()).thenReturn(TEST_DIMENSION);

        KNNMethodContext methodContext = mock(KNNMethodContext.class);
        when(methodContext.getSpaceType()).thenReturn(SpaceType.HAMMING);
        when(methodContext.getKnnEngine()).thenReturn(KNNEngine.FAISS);

        KNNLibraryIndexingContext libraryContext = mock(KNNLibraryIndexingContext.class);
        when(libraryContext.getQuantizationConfig()).thenReturn(QuantizationConfig.EMPTY);
        when(libraryContext.getLibraryParameters()).thenReturn(Collections.emptyMap());
        when(libraryContext.getResolvedSpec()).thenReturn(nonSQOneBitSpec());
        when(libraryContext.getVectorTransformer()).thenReturn(null);

        // HAMMING has no direct Lucene VectorSimilarityFunction, so the fallback to DEFAULT is exercised
        FieldTypeConfig config = FaissFieldStrategy.INSTANCE.buildFieldTypeConfig(
            mappingConfig,
            methodContext,
            libraryContext,
            VectorDataType.BINARY,
            Version.CURRENT,
            true
        );

        assertTrue(config.isUseLuceneBasedVectorField());
        assertEquals(TEST_DIMENSION / 8, config.getFieldType().vectorDimension());
        assertEquals(VectorDataType.BINARY.getValue(), config.getFieldType().getAttributes().get(VECTOR_DATA_TYPE_FIELD));
    }

    public void testBuildFieldTypeConfigForFaissWithPre30Version_thenDefaultSimilarity() {
        KNNMappingConfig mappingConfig = mock(KNNMappingConfig.class);
        when(mappingConfig.getDimension()).thenReturn(TEST_DIMENSION);

        KNNMethodContext methodContext = mock(KNNMethodContext.class);
        when(methodContext.getSpaceType()).thenReturn(SpaceType.INNER_PRODUCT);
        when(methodContext.getKnnEngine()).thenReturn(KNNEngine.FAISS);

        KNNLibraryIndexingContext libraryContext = mock(KNNLibraryIndexingContext.class);
        when(libraryContext.getQuantizationConfig()).thenReturn(QuantizationConfig.EMPTY);
        when(libraryContext.getLibraryParameters()).thenReturn(Collections.emptyMap());
        when(libraryContext.getResolvedSpec()).thenReturn(nonSQOneBitSpec());
        when(libraryContext.getVectorTransformer()).thenReturn(null);

        // Pre-3.0 versions always use the DEFAULT space type's similarity function
        FieldTypeConfig config = FaissFieldStrategy.INSTANCE.buildFieldTypeConfig(
            mappingConfig,
            methodContext,
            libraryContext,
            VectorDataType.FLOAT,
            Version.V_2_19_0,
            true
        );

        assertTrue(config.isUseLuceneBasedVectorField());
        assertEquals(
            SpaceType.DEFAULT.getKnnVectorSimilarityFunction().getVectorSimilarityFunction(),
            config.getFieldType().vectorSimilarityFunction()
        );
    }

    public void testBuildFieldTypeConfigForFaissWithNonSQOneBitSpec() {
        KNNMappingConfig mappingConfig = mock(KNNMappingConfig.class);
        when(mappingConfig.getDimension()).thenReturn(TEST_DIMENSION);

        KNNMethodContext methodContext = mock(KNNMethodContext.class);
        when(methodContext.getSpaceType()).thenReturn(SpaceType.L2);
        when(methodContext.getKnnEngine()).thenReturn(KNNEngine.FAISS);

        KNNLibraryIndexingContext libraryContext = mock(KNNLibraryIndexingContext.class);
        when(libraryContext.getQuantizationConfig()).thenReturn(QuantizationConfig.EMPTY);
        when(libraryContext.getLibraryParameters()).thenReturn(Collections.emptyMap());
        when(libraryContext.getResolvedSpec()).thenReturn(nonSQOneBitSpec());
        when(libraryContext.getVectorTransformer()).thenReturn(null);

        FieldTypeConfig config = FaissFieldStrategy.INSTANCE.buildFieldTypeConfig(
            mappingConfig,
            methodContext,
            libraryContext,
            VectorDataType.FLOAT,
            Version.CURRENT,
            true
        );

        assertNotNull(config.getFieldType());
        assertTrue(config.isUseLuceneBasedVectorField());
        assertNull(config.getFieldType().getAttributes().get(SQ_CONFIG));
    }
}

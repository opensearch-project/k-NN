/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.BytesRef;
import org.junit.Assert;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.Version;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.Explicit;
import org.opensearch.common.ValidationException;
import org.opensearch.common.settings.IndexScopedSettings;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.core.common.Strings;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.ContentPath;
import org.opensearch.index.mapper.FieldMapper;
import org.opensearch.index.mapper.Mapper;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.VectorField;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfFloatsSerializer;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.faiss.QFrameBitEncoder;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;
import org.opensearch.knn.indices.ModelUtil;

import java.io.IOException;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.stream.Collectors;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.Version.CURRENT;
import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.NMSLIB_NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.common.KNNConstants.TOP_LEVEL_PARAMETER_ENGINE;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;
import static org.opensearch.knn.index.VectorDataType.SUPPORTED_VECTOR_DATA_TYPES;

@Log4j2
public class KNNVectorFieldMapperTests extends KNNTestCase {

    private static final String TEST_INDEX_NAME = "test-index-name";
    private static final String TEST_FIELD_NAME = "test-field-name";

    private static final int TEST_DIMENSION = 17;

    private static final float TEST_VECTOR_VALUE = 1.5f;

    private static final float[] TEST_VECTOR = createInitializedFloatArray(TEST_DIMENSION, TEST_VECTOR_VALUE);

    private static final byte TEST_BYTE_VECTOR_VALUE = 10;
    private static final byte[] TEST_BYTE_VECTOR = createInitializedByteArray(TEST_DIMENSION, TEST_BYTE_VECTOR_VALUE);

    private static final BytesRef TEST_VECTOR_BYTES_REF = new BytesRef(
        KNNVectorAsCollectionOfFloatsSerializer.INSTANCE.floatToByteArray(TEST_VECTOR)
    );
    private static final BytesRef TEST_BYTE_VECTOR_BYTES_REF = new BytesRef(TEST_BYTE_VECTOR);
    private static final String DIMENSION_FIELD_NAME = "dimension";
    private static final String KNN_VECTOR_TYPE = "knn_vector";
    private static final String TYPE_FIELD_NAME = "type";

    public void testBuilder_getParameters() {
        String fieldName = "test-field-name";
        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.Builder builder = new KNNVectorFieldMapper.Builder(
            fieldName,
            modelDao,
            CURRENT,
            null,
            new OriginalMappingParameters(
                VectorDataType.DEFAULT,
                TEST_DIMENSION,
                null,
                null,
                null,
                null,
                SpaceType.UNDEFINED.getValue(),
                KNNEngine.UNDEFINED.getName()
            )
        );

        assertEquals(11, builder.getParameters().size());
        List<String> actualParams = builder.getParameters().stream().map(a -> a.name).collect(Collectors.toList());
        List<String> expectedParams = Arrays.asList(
            "store",
            "doc_values",
            DIMENSION,
            VECTOR_DATA_TYPE_FIELD,
            "meta",
            KNN_METHOD,
            MODEL_ID,
            MODE_PARAMETER,
            COMPRESSION_LEVEL_PARAMETER,
            KNNConstants.TOP_LEVEL_PARAMETER_SPACE_TYPE,
            TOP_LEVEL_PARAMETER_ENGINE
        );
        assertEquals(expectedParams, actualParams);
    }

    public void testTypeParser_build_fromKnnMethodContext() throws IOException {
        // Check that knnMethodContext takes precedent over both model and legacy
        ModelDao modelDao = mock(ModelDao.class);

        SpaceType spaceType = SpaceType.DEFAULT;
        int mRight = 17;
        int mWrong = 71;

        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, TEST_DIMENSION)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, mRight)
            .endObject()
            .endObject()
            .endObject();

        // Setup settings
        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            "test-field-name-1",
            xContentBuilderToMap(xContentBuilder),
            buildParserContext("test", settings)
        );

        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper instanceof EngineFieldMapper);
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().isPresent());
        assertEquals(spaceType, knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().get().getSpaceType());
        assertEquals(
            mRight,
            knnVectorFieldMapper.fieldType()
                .getKnnMappingConfig()
                .getKnnMethodContext()
                .get()
                .getMethodComponentContext()
                .getParameters()
                .get(METHOD_PARAMETER_M)
        );
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getModelId().isEmpty());
    }

    public void testKNNVectorFieldMapper_withBlockedKNNEngine() throws IOException {
        String fieldName = "test-field-name";
        String indexName = "test-index";

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();
        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);

        // Creating a mapping with NMSLIB before version 3.0.0 (should work)
        XContentBuilder validXContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, 128)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, KNNEngine.NMSLIB.getName()) // Using deprecated engine
            .endObject()
            .endObject();

        // Should pass for versions before 3.0.0
        KNNVectorFieldMapper.Builder builderBeforeV3 = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(validXContentBuilder),
            buildLegacyParserContext(indexName, settings, Version.V_2_9_0) // Version < 3.0.0
        );
        assertNotNull(builderBeforeV3);

        // Creating a mapping with NMSLIB on or after version 3.0.0 (should fail)
        XContentBuilder invalidXContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, 128)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, KNNEngine.NMSLIB.getName()) // Using deprecated engine
            .endObject()
            .endObject();

        // Expect an exception when trying to use NMSLIB with OpenSearch 3.0.0+
        IllegalArgumentException exception = expectThrows(
            IllegalArgumentException.class,
            () -> typeParser.parse(
                fieldName,
                xContentBuilderToMap(invalidXContentBuilder),
                buildParserContext(indexName, settings) // Version >= 3.0.0
            )
        );

        assertNotNull(exception.getMessage());

        // Creating a mapping with FAISS or LUCENE (should work)
        XContentBuilder validFaissBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, 128)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, KNNEngine.FAISS.getName()) // Valid engine
            .endObject()
            .endObject();

        KNNVectorFieldMapper.Builder builderWithFaiss = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(validFaissBuilder),
            buildParserContext(indexName, settings) // Version >= 3.0.0
        );
        assertNotNull(builderWithFaiss);
    }

    public void testKNNVectorFieldMapperLucene_docValueDefaults() throws IOException {
        String fieldName = "test-field-name";
        String indexName = "test-index";

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();
        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);

        // Creating a mapping before version 3.0.0 (doc values should be true)
        XContentBuilder legacyDocValuesContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, 128)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, KNNEngine.LUCENE.getName())
            .endObject()
            .endObject();

        // Should be true for versions before 3.0.0
        KNNVectorFieldMapper.Builder builderBeforeV3 = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(legacyDocValuesContentBuilder),
            buildLegacyParserContext(indexName, settings, Version.V_2_19_0) // Version < 3.0.0
        );
        assertNotNull(builderBeforeV3);
        assertTrue(builderBeforeV3.hasDocValues.getValue());

        // Creating a mapping with Lucene on or after version 3.0.0 (doc values should default to false)
        XContentBuilder currentDocValuesContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, 128)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, KNNEngine.LUCENE.getName())
            .endObject()
            .endObject();

        KNNVectorFieldMapper.Builder builderAfterV3 = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(currentDocValuesContentBuilder),
            buildParserContext(indexName, settings) // Version >= 3.0.0
        );
        assertNotNull(builderAfterV3);
        assertFalse(builderAfterV3.hasDocValues.getValue());
    }

    public void testKNNVectorFieldMapperModel_docValueDefaults() throws IOException {
        String fieldName = "test-field-name";
        String indexName = "test-index";
        String modelId = "test-model-id";

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();
        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);

        // Creating a model mapping before version 3.0.0 (doc values should be true)
        XContentBuilder legacyDocValuesContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(MODEL_ID, modelId)
            .endObject();

        // Should be true for versions before 3.0.0
        KNNVectorFieldMapper.Builder builderBeforeV3 = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(legacyDocValuesContentBuilder),
            buildLegacyParserContext(indexName, settings, Version.V_2_19_0) // Version < 3.0.0
        );
        assertNotNull(builderBeforeV3);
        assertTrue(builderBeforeV3.hasDocValues.getValue());

        // Creating a mapping with model on or after version 3.0.0 (doc values should default to false)
        XContentBuilder currentDocValuesContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(MODEL_ID, modelId)
            .endObject();

        KNNVectorFieldMapper.Builder builderAfterV3 = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(currentDocValuesContentBuilder),
            buildParserContext(indexName, settings) // Version >= 3.0.0
        );
        assertNotNull(builderAfterV3);
        assertFalse(builderAfterV3.hasDocValues.getValue());
    }

    public void testTypeParser_withDifferentSpaceTypeCombinations_thenSuccess() throws IOException {
        // Check that knnMethodContext takes precedent over both model and legacy
        ModelDao modelDao = mock(ModelDao.class);
        int mForSetting = 71;
        // Setup settings
        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();
        SpaceType methodSpaceType = SpaceType.COSINESIMIL;
        SpaceType topLevelSpaceType = SpaceType.INNER_PRODUCT;
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);

        // space type provided at top level but not in the method
        XContentBuilder xContentBuilder = createXContentForFieldMapping(topLevelSpaceType, null, null, null, TEST_DIMENSION);

        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            "test-field-name-1",
            xContentBuilderToMap(xContentBuilder),
            buildParserContext("test", settings)
        );

        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper instanceof EngineFieldMapper);
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().isPresent());
        assertEquals(topLevelSpaceType, knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().get().getSpaceType());
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getModelId().isEmpty());

        // not setting any space type
        xContentBuilder = createXContentForFieldMapping(null, null, null, null, TEST_DIMENSION);

        builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            "test-field-name-1",
            xContentBuilderToMap(xContentBuilder),
            buildParserContext("test", settings)
        );

        builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper instanceof EngineFieldMapper);
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().isPresent());
        assertEquals(SpaceType.DEFAULT, knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().get().getSpaceType());
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getModelId().isEmpty());

        // if space types are same
        xContentBuilder = createXContentForFieldMapping(topLevelSpaceType, topLevelSpaceType, null, null, TEST_DIMENSION);
        builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            "test-field-name-1",
            xContentBuilderToMap(xContentBuilder),
            buildParserContext("test", settings)
        );

        builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper instanceof EngineFieldMapper);
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().isPresent());
        assertEquals(topLevelSpaceType, knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().get().getSpaceType());
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getModelId().isEmpty());

        // if space types are not same
        xContentBuilder = createXContentForFieldMapping(topLevelSpaceType, methodSpaceType, null, null, TEST_DIMENSION);

        XContentBuilder finalXContentBuilder = xContentBuilder;
        Assert.assertThrows(
            MapperParsingException.class,
            () -> typeParser.parse("test-field-name-1", xContentBuilderToMap(finalXContentBuilder), buildParserContext("test", settings))
        );

        // if space types not provided and field is binary
        xContentBuilder = createXContentForFieldMapping(null, null, KNNEngine.FAISS, VectorDataType.BINARY, 8);
        builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            "test-field-name-1",
            xContentBuilderToMap(xContentBuilder),
            buildParserContext("test", settings)
        );

        builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper instanceof EngineFieldMapper);
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().isPresent());
        assertEquals(
            SpaceType.DEFAULT_BINARY,
            knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().get().getSpaceType()
        );
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getModelId().isEmpty());

        // mock useKNNMethodContextFromLegacy to simulate index ix created before 2_18
        try (MockedStatic<KNNVectorFieldMapper> utilMockedStatic = Mockito.mockStatic(KNNVectorFieldMapper.class)) {
            utilMockedStatic.when(() -> KNNVectorFieldMapper.useKNNMethodContextFromLegacy(any(), any())).thenReturn(true);
            // if space type is provided and legacy mappings is hit
            xContentBuilder = XContentFactory.jsonBuilder()
                .startObject()
                .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
                .field(DIMENSION_FIELD_NAME, TEST_DIMENSION)
                .field(KNNConstants.TOP_LEVEL_PARAMETER_SPACE_TYPE, topLevelSpaceType.getValue())
                .endObject();
            builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
                "test-field-name-1",
                xContentBuilderToMap(xContentBuilder),
                buildParserContext("test", settings)
            );

            builderContext = new Mapper.BuilderContext(settings, new ContentPath());
            knnVectorFieldMapper = builder.build(builderContext);
            assertTrue(knnVectorFieldMapper instanceof EngineFieldMapper);
            assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().isPresent());
            assertEquals(
                topLevelSpaceType,
                knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().get().getSpaceType()
            );
            // this check ensures that legacy mapping is hit, as in legacy mapping we pick M from index settings
            assertEquals(
                16,
                knnVectorFieldMapper.fieldType()
                    .getKnnMappingConfig()
                    .getKnnMethodContext()
                    .get()
                    .getMethodComponentContext()
                    .getParameters()
                    .get(METHOD_PARAMETER_M)
            );
            assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getModelId().isEmpty());
        }
    }

    public void testDifferentEngineCombinations_thenSuccess() throws IOException {
        // Setup settings
        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();
        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);

        // engine provided in method but not at the top-level
        XContentBuilder xContentBuilder = createXContentForFieldMapping_Engine(KNNEngine.LUCENE, null, null, null, TEST_DIMENSION);

        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            "test-field-name-1",
            xContentBuilderToMap(xContentBuilder),
            buildParserContext("test", settings)
        );

        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper instanceof EngineFieldMapper);
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().isPresent());
        assertEquals(KNNEngine.LUCENE, knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().get().getKnnEngine());
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getModelId().isEmpty());

        // engine provided at top level but not in the method
        xContentBuilder = createXContentForFieldMapping_Engine(null, KNNEngine.LUCENE, null, null, TEST_DIMENSION);

        builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            "test-field-name-1",
            xContentBuilderToMap(xContentBuilder),
            buildParserContext("test", settings)
        );

        builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper instanceof EngineFieldMapper);
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().isPresent());
        assertEquals(KNNEngine.LUCENE, knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().get().getKnnEngine());
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getModelId().isEmpty());

        // not setting any engine
        xContentBuilder = createXContentForFieldMapping_Engine(null, null, SpaceType.DEFAULT, null, TEST_DIMENSION);

        builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            "test-field-name-1",
            xContentBuilderToMap(xContentBuilder),
            buildParserContext("test", settings)
        );

        builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper instanceof EngineFieldMapper);
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().isPresent());
        assertEquals(KNNEngine.DEFAULT, knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().get().getKnnEngine());
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getModelId().isEmpty());

        // if engines are same
        xContentBuilder = createXContentForFieldMapping_Engine(KNNEngine.LUCENE, KNNEngine.LUCENE, null, null, TEST_DIMENSION);
        builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            "test-field-name-1",
            xContentBuilderToMap(xContentBuilder),
            buildParserContext("test", settings)
        );

        builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper instanceof EngineFieldMapper);
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().isPresent());
        assertEquals(KNNEngine.LUCENE, knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().get().getKnnEngine());
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getModelId().isEmpty());

        // if engines are not same; should throw exception
        xContentBuilder = createXContentForFieldMapping_Engine(KNNEngine.LUCENE, KNNEngine.FAISS, null, null, TEST_DIMENSION);
        XContentBuilder finalXContentBuilder_diff = xContentBuilder;
        Assert.assertThrows(
            MapperParsingException.class,
            () -> typeParser.parse(
                "test-field-name-1",
                xContentBuilderToMap(finalXContentBuilder_diff),
                buildParserContext("test", settings)
            )
        );
    }

    public void testTypeParser_withSpaceTypeAndMode_thenSuccess() throws IOException {
        // Check that knnMethodContext takes precedent over both model and legacy
        ModelDao modelDao = mock(ModelDao.class);
        // Setup settings
        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        SpaceType topLevelSpaceType = SpaceType.INNER_PRODUCT;
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION, TEST_DIMENSION)
            .field(MODE_PARAMETER, Mode.ON_DISK.getName())
            .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x16.getName())
            .field(KNNConstants.TOP_LEVEL_PARAMETER_SPACE_TYPE, topLevelSpaceType.getValue())
            .endObject();
        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            "test-field-name-1",
            xContentBuilderToMap(xContentBuilder),
            buildParserContext("test", settings)
        );

        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper instanceof EngineFieldMapper);
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().isPresent());
        assertEquals(topLevelSpaceType, knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().get().getSpaceType());
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getModelId().isEmpty());
    }

    public void testBuilder_build_fromModel() {
        // Check that modelContext takes precedent over legacy
        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.Builder builder = new KNNVectorFieldMapper.Builder("test-field-name-1", modelDao, CURRENT, null, null);

        SpaceType spaceType = SpaceType.COSINESIMIL;
        int m = 17;
        int efConstruction = 17;

        // Setup settings
        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        String modelId = "Random modelId";
        ModelMetadata mockedModelMetadata = new ModelMetadata(
            KNNEngine.FAISS,
            SpaceType.L2,
            129,
            ModelState.CREATED,
            ZonedDateTime.now(ZoneOffset.UTC).toString(),
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.FLOAT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.V_EMPTY
        );
        builder.modelId.setValue(modelId);
        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());

        when(modelDao.getMetadata(modelId)).thenReturn(mockedModelMetadata);
        builder.setOriginalParameters(new OriginalMappingParameters(builder));
        KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper instanceof ModelFieldMapper);
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getModelId().isPresent());
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().isEmpty());
    }

    public void testSpaceType_build_fromLegacy() throws IOException {
        // Check legacy is picked up if model context and method context are not set
        ModelDao modelDao = mock(ModelDao.class);
        int m = 17;
        int efConstruction = 17;
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, 12)
            .endObject();

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            "test-field-name-1",
            xContentBuilderToMap(xContentBuilder),
            buildLegacyParserContext("test", settings, Version.V_2_15_0)
        );

        // Setup settings
        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper instanceof EngineFieldMapper);
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().isPresent());
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getModelId().isEmpty());
        assertEquals(SpaceType.L2, knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().get().getSpaceType());
    }

    public void testBuilder_build_fromLegacy() throws IOException {
        // Check legacy is picked up if model context and method context are not set
        ModelDao modelDao = mock(ModelDao.class);
        int m = 17;
        int efConstruction = 17;
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, 12)
            .endObject();

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            "test-field-name-1",
            xContentBuilderToMap(xContentBuilder),
            buildParserContext("test", settings)
        );

        // Setup settings
        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper instanceof EngineFieldMapper);
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().isPresent());
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getModelId().isEmpty());
        assertEquals(SpaceType.L2, knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().get().getSpaceType());
    }

    public void testBuilder_parse_fromKnnMethodContext_luceneEngine() throws IOException {
        String fieldName = "test-field-name";
        String indexName = "test-index-name";

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);

        KNNEngine.LUCENE.setInitialized(false);

        int efConstruction = 321;
        int m = 12;
        int dimension = 133;
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2)
            .field(KNN_ENGINE, LUCENE_NAME)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction)
            .field(METHOD_PARAMETER_M, m)
            .endObject()
            .endObject()
            .endObject();
        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(xContentBuilder),
            buildParserContext(indexName, settings)
        );
        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        builder.build(builderContext);

        assertEquals(METHOD_HNSW, builder.knnMethodContext.get().getMethodComponentContext().getName());
        assertEquals(
            efConstruction,
            builder.knnMethodContext.get().getMethodComponentContext().getParameters().get(METHOD_PARAMETER_EF_CONSTRUCTION)
        );
        assertTrue(KNNEngine.LUCENE.isInitialized());

        XContentBuilder xContentBuilderEmptyParams = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2)
            .field(KNN_ENGINE, LUCENE_NAME)
            .endObject()
            .endObject();
        KNNVectorFieldMapper.Builder builderEmptyParams = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(xContentBuilderEmptyParams),
            buildParserContext(indexName, settings)
        );

        assertEquals(METHOD_HNSW, builder.knnMethodContext.get().getMethodComponentContext().getName());
        assertTrue(builderEmptyParams.knnMethodContext.get().getMethodComponentContext().getParameters().isEmpty());

        XContentBuilder xContentBuilderUnsupportedParam = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2)
            .field(KNN_ENGINE, LUCENE_NAME)
            .startObject(PARAMETERS)
            .field("RANDOM_PARAM", 0)
            .endObject()
            .endObject()
            .endObject();

        expectThrows(
            ValidationException.class,
            () -> typeParser.parse(
                fieldName,
                xContentBuilderToMap(xContentBuilderUnsupportedParam),
                buildParserContext(indexName, settings)
            )
        );
    }

    public void testTypeParser_parse_fromKnnMethodContext_invalidDimension() throws IOException {
        String fieldName = "test-field-name";
        String indexName = "test-index-name";

        Settings settings = Settings.builder().put(settings(CURRENT).put(KNN_INDEX, true).build()).build();

        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);

        int efConstruction = 321;

        XContentBuilder xContentBuilderOverMaxDimension = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, 20000)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2)
            .field(KNN_ENGINE, LUCENE_NAME)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction)
            .endObject()
            .endObject()
            .endObject();

        IllegalArgumentException ex = expectThrows(
            IllegalArgumentException.class,
            () -> typeParser.parse(
                fieldName,
                xContentBuilderToMap(xContentBuilderOverMaxDimension),
                buildParserContext(indexName, settings)
            )
        );
        assertTrue(ex.getMessage().contains("Dimension value cannot be greater than 16000 for vector with engine: lucene"));

        XContentBuilder xContentBuilderInvalidDimension = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, "2147483648")
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2)
            .field(KNN_ENGINE, NMSLIB_NAME)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction)
            .endObject()
            .endObject()
            .endObject();

        IllegalArgumentException exInvalidDimension = expectThrows(
            IllegalArgumentException.class,
            () -> typeParser.parse(
                fieldName,
                xContentBuilderToMap(xContentBuilderInvalidDimension),
                buildParserContext(indexName, settings)
            )
        );
        assertEquals(
            "Unable to parse [dimension] from provided value [2147483648] for vector [test-field-name]",
            exInvalidDimension.getMessage()
        );
    }

    @SneakyThrows
    public void testTypeParser_parse_compressionAndModeParameter() {
        String fieldName = "test-field-name-vec";
        String indexName = "test-index-name-vec";

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);

        XContentBuilder xContentBuilder1 = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION, 10)
            .field(VECTOR_DATA_TYPE_FIELD, VectorDataType.DEFAULT.getValue())
            .field(MODE_PARAMETER, Mode.ON_DISK.getName())
            .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x16.getName())
            .endObject();

        Mapper.Builder<?> builder = typeParser.parse(
            fieldName,
            xContentBuilderToMap(xContentBuilder1),
            buildParserContext(indexName, settings)
        );

        assertTrue(builder instanceof KNNVectorFieldMapper.Builder);
        assertEquals(Mode.ON_DISK.getName(), ((KNNVectorFieldMapper.Builder) builder).mode.get());
        assertEquals(CompressionLevel.x16.getName(), ((KNNVectorFieldMapper.Builder) builder).compressionLevel.get());

        XContentBuilder xContentBuilder2 = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION, 10)
            .field(VECTOR_DATA_TYPE_FIELD, VectorDataType.DEFAULT.getValue())
            .field(MODE_PARAMETER, "invalid")
            .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x16.getName())
            .endObject();

        expectThrows(
            MapperParsingException.class,
            () -> typeParser.parse(fieldName, xContentBuilderToMap(xContentBuilder2), buildParserContext(indexName, settings))
        );

        XContentBuilder xContentBuilder3 = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION, 10)
            .field(VECTOR_DATA_TYPE_FIELD, VectorDataType.DEFAULT.getValue())
            .field(COMPRESSION_LEVEL_PARAMETER, "invalid")
            .endObject();

        expectThrows(
            MapperParsingException.class,
            () -> typeParser.parse(fieldName, xContentBuilderToMap(xContentBuilder3), buildParserContext(indexName, settings))
        );

        XContentBuilder xContentBuilder4 = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(VECTOR_DATA_TYPE_FIELD, VectorDataType.DEFAULT.getValue())
            .field(MODEL_ID, "test")
            .field(MODE_PARAMETER, Mode.ON_DISK.getName())
            .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x16.getName())
            .endObject();

        expectThrows(
            MapperParsingException.class,
            () -> typeParser.parse(fieldName, xContentBuilderToMap(xContentBuilder4), buildParserContext(indexName, settings))
        );
    }

    // Validate TypeParser parsing invalid vector data_type which throws exception
    @SneakyThrows
    public void testTypeParser_parse_invalidVectorDataType() {
        String fieldName = "test-field-name-vec";
        String indexName = "test-index-name-vec";
        String vectorDataType = "invalid";

        Settings settings = Settings.builder().put(settings(CURRENT).build()).build();

        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);

        XContentBuilder xContentBuilderOverInvalidVectorType = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION, 10)
            .field(VECTOR_DATA_TYPE_FIELD, vectorDataType)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2)
            .field(KNN_ENGINE, LUCENE_NAME)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, 128)
            .endObject()
            .endObject()
            .endObject();

        IllegalArgumentException ex = expectThrows(
            IllegalArgumentException.class,
            () -> typeParser.parse(
                fieldName,
                xContentBuilderToMap(xContentBuilderOverInvalidVectorType),
                buildParserContext(indexName, settings)
            )
        );
        assertEquals(
            String.format(
                Locale.ROOT,
                "Invalid value provided for [%s] field. Supported values are [%s]",
                VECTOR_DATA_TYPE_FIELD,
                SUPPORTED_VECTOR_DATA_TYPES
            ),
            ex.getMessage()
        );
    }

    public void testTypeParser_parse_fromKnnMethodContext_invalidSpaceType() throws IOException {
        String fieldName = "test-field-name";
        String indexName = "test-index-name";

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);

        int efConstruction = 321;
        int dimension = 133;
        XContentBuilder xContentBuilderL1SpaceType = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L1.getValue())
            .field(KNN_ENGINE, LUCENE_NAME)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction)
            .endObject()
            .endObject()
            .endObject();

        expectThrows(
            ValidationException.class,
            () -> typeParser.parse(fieldName, xContentBuilderToMap(xContentBuilderL1SpaceType), buildParserContext(indexName, settings))
        );
    }

    public void testTypeParser_parse_fromKnnMethodContext() throws IOException {
        // Check that knnMethodContext is set
        String fieldName = "test-field-name";
        String indexName = "test-index-name";

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);

        int efConstruction = 321;
        int dimension = 133;
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction)
            .endObject()
            .endObject()
            .endObject();

        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(xContentBuilder),
            buildParserContext(indexName, settings)
        );

        assertEquals(METHOD_HNSW, builder.knnMethodContext.get().getMethodComponentContext().getName());
        assertEquals(
            efConstruction,
            builder.knnMethodContext.get().getMethodComponentContext().getParameters().get(METHOD_PARAMETER_EF_CONSTRUCTION)
        );

        // Test invalid parameter
        XContentBuilder xContentBuilder2 = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .startObject(PARAMETERS)
            .field("invalid", "invalid")
            .endObject()
            .endObject()
            .endObject();

        expectThrows(
            ValidationException.class,
            () -> typeParser.parse(fieldName, xContentBuilderToMap(xContentBuilder2), buildParserContext(indexName, settings))
        );

        // Test invalid method
        XContentBuilder xContentBuilder3 = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, "invalid")
            .endObject()
            .endObject();

        expectThrows(
            IllegalArgumentException.class,
            () -> typeParser.parse(fieldName, xContentBuilderToMap(xContentBuilder3), buildParserContext(indexName, settings))
        );

        // Test missing required parameter: dimension
        XContentBuilder xContentBuilder4 = XContentFactory.jsonBuilder().startObject().field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE).endObject();

        expectThrows(
            IllegalArgumentException.class,
            () -> typeParser.parse(fieldName, xContentBuilderToMap(xContentBuilder4), buildParserContext(indexName, settings))
        );

        // Check that this fails if model id is also set
        XContentBuilder xContentBuilder5 = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .field(MODEL_ID, "test-id")
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .startObject(PARAMETERS)
            .field("invalid", "invalid")
            .endObject()
            .endObject()
            .endObject();

        expectThrows(
            IllegalArgumentException.class,
            () -> typeParser.parse(fieldName, xContentBuilderToMap(xContentBuilder5), buildParserContext(indexName, settings))
        );
    }

    public void testTypeParser_parse_fromModel() throws IOException {
        // Check that modelContext is set for the builder
        String fieldName = "test-field-name";
        String indexName = "test-index-name";

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);

        String modelId = "test-id";
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(MODEL_ID, modelId)
            .endObject();

        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(xContentBuilder),
            buildParserContext(indexName, settings)
        );

        assertEquals(modelId, builder.modelId.get());

        // Fails if both model_id and dimension are set after 2.19
        XContentBuilder xContentBuilder2 = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, TEST_DIMENSION)
            .field(MODEL_ID, modelId)
            .endObject();

        expectThrows(
            IllegalArgumentException.class,
            () -> typeParser.parse(fieldName, xContentBuilderToMap(xContentBuilder2), buildParserContext(indexName, settings))
        );

        // Should not fail if both are set before 2.19
        typeParser.parse(
            fieldName,
            xContentBuilderToMap(xContentBuilder2),
            buildLegacyParserContext(indexName, settings, Version.V_2_18_1)
        );

    }

    public void testTypeParser_parse_fromLegacy() throws IOException {
        // Check that the particular values are set in builder
        String fieldName = "test-field-name";
        String indexName = "test-index-name";

        int m = 144;
        int efConstruction = 123;
        SpaceType spaceType = SpaceType.L2;
        Settings settings = Settings.builder().put(settings(CURRENT).build()).build();

        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);

        int dimension = 122;
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .endObject();

        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(xContentBuilder),
            buildParserContext(indexName, settings)
        );

        assertNull(builder.modelId.get());
        assertNull(builder.knnMethodContext.get());
    }

    public void testKNNVectorFieldMapperMerge_whenModeAndCompressionIsPresent_thenSuccess() throws IOException {
        String fieldName = "test-field-name";
        String indexName = "test-index-name";

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();
        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);

        int dimension = 133;
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .field(MODE_PARAMETER, Mode.ON_DISK.getName())
            .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x32.getName())
            .endObject();

        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(xContentBuilder),
            buildParserContext(indexName, settings)
        );
        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper1 = builder.build(builderContext);

        // merge with itself - should be successful
        KNNVectorFieldMapper knnVectorFieldMapperMerge1 = (KNNVectorFieldMapper) knnVectorFieldMapper1.merge(knnVectorFieldMapper1);
        assertEquals(
            knnVectorFieldMapper1.fieldType().getKnnMappingConfig().getKnnMethodContext().get(),
            knnVectorFieldMapperMerge1.fieldType().getKnnMappingConfig().getKnnMethodContext().get()
        );

        assertEquals(
            knnVectorFieldMapper1.fieldType().getKnnMappingConfig().getCompressionLevel(),
            knnVectorFieldMapperMerge1.fieldType().getKnnMappingConfig().getCompressionLevel()
        );
        assertEquals(
            knnVectorFieldMapper1.fieldType().getKnnMappingConfig().getMode(),
            knnVectorFieldMapperMerge1.fieldType().getKnnMappingConfig().getMode()
        );

        // merge with another mapper of the same field with same context
        KNNVectorFieldMapper knnVectorFieldMapper2 = builder.build(builderContext);
        KNNVectorFieldMapper knnVectorFieldMapperMerge2 = (KNNVectorFieldMapper) knnVectorFieldMapper1.merge(knnVectorFieldMapper2);
        assertEquals(
            knnVectorFieldMapper1.fieldType().getKnnMappingConfig().getKnnMethodContext().get(),
            knnVectorFieldMapperMerge2.fieldType().getKnnMappingConfig().getKnnMethodContext().get()
        );

        assertEquals(
            knnVectorFieldMapper1.fieldType().getKnnMappingConfig().getCompressionLevel(),
            knnVectorFieldMapperMerge2.fieldType().getKnnMappingConfig().getCompressionLevel()
        );
        assertEquals(
            knnVectorFieldMapper1.fieldType().getKnnMappingConfig().getMode(),
            knnVectorFieldMapperMerge2.fieldType().getKnnMappingConfig().getMode()
        );
    }

    public void testKNNVectorFieldMapper_merge_fromKnnMethodContext() throws IOException {
        String fieldName = "test-field-name";
        String indexName = "test-index-name";

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);

        int dimension = 133;
        int efConstruction = 321;
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction)
            .endObject()
            .endObject()
            .endObject();

        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(xContentBuilder),
            buildParserContext(indexName, settings)
        );

        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper1 = builder.build(builderContext);

        // merge with itself - should be successful
        KNNVectorFieldMapper knnVectorFieldMapperMerge1 = (KNNVectorFieldMapper) knnVectorFieldMapper1.merge(knnVectorFieldMapper1);
        assertEquals(
            knnVectorFieldMapper1.fieldType().getKnnMappingConfig().getKnnMethodContext().get(),
            knnVectorFieldMapperMerge1.fieldType().getKnnMappingConfig().getKnnMethodContext().get()
        );

        // merge with another mapper of the same field with same context
        KNNVectorFieldMapper knnVectorFieldMapper2 = builder.build(builderContext);
        KNNVectorFieldMapper knnVectorFieldMapperMerge2 = (KNNVectorFieldMapper) knnVectorFieldMapper1.merge(knnVectorFieldMapper2);
        assertEquals(
            knnVectorFieldMapper1.fieldType().getKnnMappingConfig().getKnnMethodContext().get(),
            knnVectorFieldMapperMerge2.fieldType().getKnnMappingConfig().getKnnMethodContext().get()
        );

        // merge with another mapper of the same field with different context
        xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .endObject()
            .endObject();

        builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(xContentBuilder),
            buildParserContext(indexName, settings)
        );
        KNNVectorFieldMapper knnVectorFieldMapper3 = builder.build(builderContext);
        expectThrows(IllegalArgumentException.class, () -> knnVectorFieldMapper1.merge(knnVectorFieldMapper3));
    }

    public void testKNNVectorFieldMapper_merge_fromModel() throws IOException {
        String fieldName = "test-field-name";
        String indexName = "test-index-name";

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        String modelId = "test-id";
        int dimension = 133;

        ModelDao mockModelDao = mock(ModelDao.class);
        ModelMetadata mockModelMetadata = new ModelMetadata(
            KNNEngine.DEFAULT,
            SpaceType.DEFAULT,
            dimension,
            ModelState.CREATED,
            ZonedDateTime.now(ZoneOffset.UTC).toString(),
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.FLOAT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.V_EMPTY
        );
        when(mockModelDao.getMetadata(modelId)).thenReturn(mockModelMetadata);

        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> mockModelDao);

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(MODEL_ID, modelId)
            .endObject();

        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(xContentBuilder),
            buildParserContext(indexName, settings)
        );

        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper1 = builder.build(builderContext);

        // merge with itself - should be successful
        KNNVectorFieldMapper knnVectorFieldMapperMerge1 = (KNNVectorFieldMapper) knnVectorFieldMapper1.merge(knnVectorFieldMapper1);
        assertEquals(
            knnVectorFieldMapper1.fieldType().getKnnMappingConfig().getModelId().get(),
            knnVectorFieldMapperMerge1.fieldType().getKnnMappingConfig().getModelId().get()
        );

        // merge with another mapper of the same field with same context
        KNNVectorFieldMapper knnVectorFieldMapper2 = builder.build(builderContext);
        KNNVectorFieldMapper knnVectorFieldMapperMerge2 = (KNNVectorFieldMapper) knnVectorFieldMapper1.merge(knnVectorFieldMapper2);
        assertEquals(
            knnVectorFieldMapper1.fieldType().getKnnMappingConfig().getModelId().get(),
            knnVectorFieldMapperMerge2.fieldType().getKnnMappingConfig().getModelId().get()
        );

        // merge with another mapper of the same field with different context
        xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .endObject()
            .endObject();

        builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(xContentBuilder),
            buildParserContext(indexName, settings)
        );

        KNNVectorFieldMapper knnVectorFieldMapper3 = builder.build(builderContext);
        expectThrows(IllegalArgumentException.class, () -> knnVectorFieldMapper1.merge(knnVectorFieldMapper3));
    }

    @SneakyThrows
    public void testMethodFieldMapper_saveBestMatchingVectorSimilarityFunction() {
        final MethodComponentContext methodComponentContext = new MethodComponentContext(METHOD_HNSW, Collections.emptyMap());

        doTestMethodFieldMapper_saveBestMatchingVectorSimilarityFunction(methodComponentContext, SpaceType.INNER_PRODUCT, Version.V_2_19_0);

        doTestMethodFieldMapper_saveBestMatchingVectorSimilarityFunction(methodComponentContext, SpaceType.HAMMING, Version.V_2_19_0);

        doTestMethodFieldMapper_saveBestMatchingVectorSimilarityFunction(methodComponentContext, SpaceType.INNER_PRODUCT, Version.V_3_0_0);

        doTestMethodFieldMapper_saveBestMatchingVectorSimilarityFunction(methodComponentContext, SpaceType.HAMMING, Version.V_3_0_0);
    }

    @SneakyThrows
    private void doTestMethodFieldMapper_saveBestMatchingVectorSimilarityFunction(
        MethodComponentContext methodComponentContext,
        SpaceType spaceType,
        final Version version
    ) {
        try (MockedStatic<KNNVectorFieldMapperUtil> utilMockedStatic = Mockito.mockStatic(KNNVectorFieldMapperUtil.class)) {
            final VectorDataType dataType = VectorDataType.FLOAT;
            final int dimension = TEST_VECTOR.length;

            final KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
                .vectorDataType(dataType)
                .versionCreated(version)
                .dimension(dimension)
                .build();
            final KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.FAISS, spaceType, methodComponentContext);

            final IndexSettings indexSettingsMock = mock(IndexSettings.class);
            when(indexSettingsMock.getSettings()).thenReturn(Settings.EMPTY);
            final ParseContext.Document document = new ParseContext.Document();
            final ContentPath contentPath = new ContentPath();
            final ParseContext parseContext = mock(ParseContext.class);
            when(parseContext.doc()).thenReturn(document);
            when(parseContext.path()).thenReturn(contentPath);
            when(parseContext.parser()).thenReturn(createXContentParser(dataType));
            when(parseContext.indexSettings()).thenReturn(indexSettingsMock);

            utilMockedStatic.when(() -> KNNVectorFieldMapperUtil.useLuceneKNNVectorsFormat(Mockito.any())).thenReturn(true);
            utilMockedStatic.when(() -> KNNVectorFieldMapperUtil.useFullFieldNameValidation(Mockito.any())).thenReturn(true);

            final OriginalMappingParameters originalMappingParameters = new OriginalMappingParameters(
                dataType,
                dimension,
                knnMethodContext,
                Mode.NOT_CONFIGURED.getName(),
                CompressionLevel.NOT_CONFIGURED.getName(),
                null,
                SpaceType.UNDEFINED.getValue(),
                KNNEngine.UNDEFINED.getName()
            );
            originalMappingParameters.setResolvedKnnMethodContext(knnMethodContext);
            EngineFieldMapper methodFieldMapper = EngineFieldMapper.createFieldMapper(
                TEST_FIELD_NAME,
                TEST_FIELD_NAME,
                Collections.emptyMap(),
                knnMethodConfigContext,
                FieldMapper.MultiFields.empty(),
                FieldMapper.CopyTo.empty(),
                new Explicit<>(true, true),
                false,
                false,
                originalMappingParameters
            );
            methodFieldMapper.parseCreateField(parseContext, dimension, dataType);
            final IndexableField field1 = document.getFields().get(0);

            VectorSimilarityFunction similarityFunction = SpaceType.DEFAULT.getKnnVectorSimilarityFunction().getVectorSimilarityFunction();

            if (version.onOrAfter(Version.V_3_0_0)) {
                // If version >= 3.0, then it should find the best matching function.
                try {
                    similarityFunction = spaceType.getKnnVectorSimilarityFunction().getVectorSimilarityFunction();
                } catch (Exception e) {
                    // Ignore
                }
            }

            assertEquals(similarityFunction, field1.fieldType().vectorSimilarityFunction());
        }
    }

    @SneakyThrows
    public void testMethodFieldMapperParseCreateField_validInput_thenDifferentFieldTypes() {
        try (MockedStatic<KNNVectorFieldMapperUtil> utilMockedStatic = Mockito.mockStatic(KNNVectorFieldMapperUtil.class)) {
            for (VectorDataType dataType : VectorDataType.values()) {
                log.info("Vector Data Type is : {}", dataType);
                int dimension = adjustDimensionForIndexing(TEST_DIMENSION, dataType);
                final MethodComponentContext methodComponentContext = new MethodComponentContext(METHOD_HNSW, Collections.emptyMap());
                SpaceType spaceType = VectorDataType.BINARY == dataType ? SpaceType.DEFAULT_BINARY : SpaceType.INNER_PRODUCT;
                KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
                    .vectorDataType(dataType)
                    .versionCreated(CURRENT)
                    .dimension(dimension)
                    .build();
                final KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.FAISS, spaceType, methodComponentContext);

                IndexSettings indexSettingsMock = mock(IndexSettings.class);
                when(indexSettingsMock.getSettings()).thenReturn(Settings.EMPTY);
                ParseContext.Document document = new ParseContext.Document();
                ContentPath contentPath = new ContentPath();
                ParseContext parseContext = mock(ParseContext.class);
                when(parseContext.doc()).thenReturn(document);
                when(parseContext.path()).thenReturn(contentPath);
                when(parseContext.parser()).thenReturn(createXContentParser(dataType));
                when(parseContext.indexSettings()).thenReturn(indexSettingsMock);

                utilMockedStatic.when(() -> KNNVectorFieldMapperUtil.useLuceneKNNVectorsFormat(Mockito.any())).thenReturn(true);
                utilMockedStatic.when(() -> KNNVectorFieldMapperUtil.useFullFieldNameValidation(Mockito.any())).thenReturn(true);

                OriginalMappingParameters originalMappingParameters = new OriginalMappingParameters(
                    dataType,
                    dimension,
                    knnMethodContext,
                    Mode.NOT_CONFIGURED.getName(),
                    CompressionLevel.NOT_CONFIGURED.getName(),
                    null,
                    SpaceType.UNDEFINED.getValue(),
                    KNNEngine.UNDEFINED.getName()
                );
                originalMappingParameters.setResolvedKnnMethodContext(knnMethodContext);
                EngineFieldMapper methodFieldMapper = EngineFieldMapper.createFieldMapper(
                    TEST_FIELD_NAME,
                    TEST_FIELD_NAME,
                    Collections.emptyMap(),
                    knnMethodConfigContext,
                    FieldMapper.MultiFields.empty(),
                    FieldMapper.CopyTo.empty(),
                    new Explicit<>(true, true),
                    false,
                    false,
                    originalMappingParameters
                );
                methodFieldMapper.parseCreateField(parseContext, dimension, dataType);

                List<IndexableField> fields = document.getFields();
                assertEquals(1, fields.size());
                IndexableField field1 = fields.get(0);
                if (dataType == VectorDataType.FLOAT) {
                    assertTrue(field1 instanceof KnnFloatVectorField);
                    assertEquals(field1.fieldType().vectorEncoding(), VectorEncoding.FLOAT32);
                } else {
                    assertTrue(field1 instanceof KnnByteVectorField);
                    assertEquals(field1.fieldType().vectorEncoding(), VectorEncoding.BYTE);
                }

                assertEquals(field1.fieldType().vectorDimension(), adjustDimensionForSearch(dimension, dataType));
                assertEquals(Integer.parseInt(field1.fieldType().getAttributes().get(DIMENSION_FIELD_NAME)), dimension);
                final VectorSimilarityFunction similarityFunction = spaceType != SpaceType.HAMMING
                    ? spaceType.getKnnVectorSimilarityFunction().getVectorSimilarityFunction()
                    : SpaceType.DEFAULT.getKnnVectorSimilarityFunction().getVectorSimilarityFunction();
                assertEquals(field1.fieldType().vectorSimilarityFunction(), similarityFunction);

                utilMockedStatic.when(() -> KNNVectorFieldMapperUtil.useLuceneKNNVectorsFormat(Mockito.any())).thenReturn(false);
                utilMockedStatic.when(() -> KNNVectorFieldMapperUtil.useFullFieldNameValidation(Mockito.any())).thenReturn(false);

                document = new ParseContext.Document();
                contentPath = new ContentPath();
                when(parseContext.doc()).thenReturn(document);
                when(parseContext.path()).thenReturn(contentPath);
                when(parseContext.parser()).thenReturn(createXContentParser(dataType));
                when(parseContext.indexSettings()).thenReturn(indexSettingsMock);
                methodFieldMapper = EngineFieldMapper.createFieldMapper(
                    TEST_FIELD_NAME,
                    TEST_FIELD_NAME,
                    Collections.emptyMap(),
                    knnMethodConfigContext,
                    FieldMapper.MultiFields.empty(),
                    FieldMapper.CopyTo.empty(),
                    new Explicit<>(true, true),
                    false,
                    false,
                    originalMappingParameters
                );

                methodFieldMapper.parseCreateField(parseContext, dimension, dataType);
                fields = document.getFields();
                assertEquals(1, fields.size());
                field1 = fields.get(0);
                assertTrue(field1 instanceof VectorField);
                assertEquals(Integer.parseInt(field1.fieldType().getAttributes().get(DIMENSION_FIELD_NAME)), dimension);
            }
        }
    }

    @SneakyThrows
    public void testModelFieldMapperParseCreateField_validInput_thenDifferentFieldTypes() {
        ModelDao modelDao = mock(ModelDao.class);
        ModelMetadata modelMetadata = mock(ModelMetadata.class);
        final MethodComponentContext methodComponentContext = new MethodComponentContext(METHOD_IVF, Collections.emptyMap());
        try (
            MockedStatic<KNNVectorFieldMapperUtil> utilMockedStatic = Mockito.mockStatic(KNNVectorFieldMapperUtil.class);
            MockedStatic<ModelUtil> modelUtilMockedStatic = Mockito.mockStatic(ModelUtil.class)
        ) {
            KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
                .vectorDataType(VectorDataType.FLOAT)
                .versionCreated(CURRENT)
                .dimension(TEST_DIMENSION)
                .build();

            for (VectorDataType dataType : VectorDataType.values()) {
                log.info("Vector Data Type is : {}", dataType);
                SpaceType spaceType = VectorDataType.BINARY == dataType ? SpaceType.DEFAULT_BINARY : SpaceType.INNER_PRODUCT;
                int dimension = adjustDimensionForIndexing(TEST_DIMENSION, dataType);
                when(modelDao.getMetadata(MODEL_ID)).thenReturn(modelMetadata);
                modelUtilMockedStatic.when(() -> ModelUtil.isModelCreated(modelMetadata)).thenReturn(true);
                when(modelMetadata.getDimension()).thenReturn(dimension);
                when(modelMetadata.getVectorDataType()).thenReturn(dataType);
                when(modelMetadata.getSpaceType()).thenReturn(spaceType);
                when(modelMetadata.getKnnEngine()).thenReturn(KNNEngine.FAISS);
                when(modelMetadata.getMethodComponentContext()).thenReturn(methodComponentContext);

                ParseContext.Document document = new ParseContext.Document();
                ContentPath contentPath = new ContentPath();
                IndexSettings indexSettingsMock = mock(IndexSettings.class);
                when(indexSettingsMock.getSettings()).thenReturn(Settings.EMPTY);

                ParseContext parseContext = mock(ParseContext.class);
                when(parseContext.doc()).thenReturn(document);
                when(parseContext.path()).thenReturn(contentPath);
                when(parseContext.parser()).thenReturn(createXContentParser(dataType));
                when(parseContext.indexSettings()).thenReturn(indexSettingsMock);

                utilMockedStatic.when(() -> KNNVectorFieldMapperUtil.useLuceneKNNVectorsFormat(Mockito.any())).thenReturn(true);
                utilMockedStatic.when(() -> KNNVectorFieldMapperUtil.useFullFieldNameValidation(Mockito.any())).thenReturn(true);

                OriginalMappingParameters originalMappingParameters = new OriginalMappingParameters(
                    VectorDataType.DEFAULT,
                    -1,
                    null,
                    Mode.NOT_CONFIGURED.getName(),
                    CompressionLevel.NOT_CONFIGURED.getName(),
                    MODEL_ID,
                    SpaceType.UNDEFINED.getValue(),
                    KNNEngine.UNDEFINED.getName()
                );

                ModelFieldMapper modelFieldMapper = ModelFieldMapper.createFieldMapper(
                    TEST_FIELD_NAME,
                    TEST_FIELD_NAME,
                    Collections.emptyMap(),
                    dataType,
                    FieldMapper.MultiFields.empty(),
                    FieldMapper.CopyTo.empty(),
                    new Explicit<>(true, true),
                    false,
                    false,
                    modelDao,
                    CURRENT,
                    originalMappingParameters,
                    knnMethodConfigContext
                );

                modelFieldMapper.parseCreateField(parseContext);

                List<IndexableField> fields = document.getFields();
                assertEquals(1, fields.size());
                IndexableField field1 = fields.get(0);
                if (dataType == VectorDataType.FLOAT) {
                    assertTrue(field1 instanceof KnnFloatVectorField);
                    assertEquals(field1.fieldType().vectorEncoding(), VectorEncoding.FLOAT32);
                } else {
                    assertTrue(field1 instanceof KnnByteVectorField);
                    assertEquals(field1.fieldType().vectorEncoding(), VectorEncoding.BYTE);
                }

                assertEquals(field1.fieldType().vectorDimension(), adjustDimensionForSearch(dimension, dataType));
                assertEquals(
                    field1.fieldType().vectorSimilarityFunction(),
                    SpaceType.DEFAULT.getKnnVectorSimilarityFunction().getVectorSimilarityFunction()
                );

                utilMockedStatic.when(() -> KNNVectorFieldMapperUtil.useLuceneKNNVectorsFormat(Mockito.any())).thenReturn(false);
                utilMockedStatic.when(() -> KNNVectorFieldMapperUtil.useFullFieldNameValidation(Mockito.any())).thenReturn(true);

                document = new ParseContext.Document();
                contentPath = new ContentPath();
                when(parseContext.doc()).thenReturn(document);
                when(parseContext.path()).thenReturn(contentPath);
                when(parseContext.parser()).thenReturn(createXContentParser(dataType));
                when(parseContext.indexSettings()).thenReturn(indexSettingsMock);
                modelFieldMapper = ModelFieldMapper.createFieldMapper(
                    TEST_FIELD_NAME,
                    TEST_FIELD_NAME,
                    Collections.emptyMap(),
                    dataType,
                    FieldMapper.MultiFields.empty(),
                    FieldMapper.CopyTo.empty(),
                    new Explicit<>(true, true),
                    false,
                    false,
                    modelDao,
                    CURRENT,
                    originalMappingParameters,
                    knnMethodConfigContext
                );

                modelFieldMapper.parseCreateField(parseContext);
                fields = document.getFields();
                assertEquals(1, fields.size());
                field1 = fields.get(0);
                assertTrue(field1 instanceof VectorField);
            }
        }
    }

    @SneakyThrows
    public void testLuceneFieldMapper_parseCreateField_docValues_withFloats() {
        IndexSettings indexSettingsMock = mock(IndexSettings.class);
        when(indexSettingsMock.getSettings()).thenReturn(Settings.EMPTY);
        ParseContext.Document document = new ParseContext.Document();
        ContentPath contentPath = new ContentPath();
        ParseContext parseContext = mock(ParseContext.class);
        when(parseContext.doc()).thenReturn(document);
        when(parseContext.path()).thenReturn(contentPath);
        when(parseContext.parser()).thenReturn(createXContentParser(VectorDataType.FLOAT));
        when(parseContext.indexSettings()).thenReturn(indexSettingsMock);
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .vectorDataType(VectorDataType.FLOAT)
            .versionCreated(CURRENT)
            .dimension(TEST_DIMENSION)
            .build();

        KNNMethodContext luceneMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.DEFAULT,
            new MethodComponentContext(METHOD_HNSW, Collections.emptyMap())
        );

        OriginalMappingParameters originalMappingParameters = new OriginalMappingParameters(
            VectorDataType.FLOAT,
            TEST_DIMENSION,
            luceneMethodContext,
            Mode.NOT_CONFIGURED.getName(),
            CompressionLevel.NOT_CONFIGURED.getName(),
            null,
            SpaceType.UNDEFINED.getValue(),
            KNNEngine.UNDEFINED.getName()
        );
        originalMappingParameters.setResolvedKnnMethodContext(originalMappingParameters.getKnnMethodContext());

        EngineFieldMapper luceneFieldMapper = EngineFieldMapper.createFieldMapper(
            TEST_FIELD_NAME,
            TEST_FIELD_NAME,
            Collections.emptyMap(),
            knnMethodConfigContext,
            FieldMapper.MultiFields.empty(),
            FieldMapper.CopyTo.empty(),
            new Explicit<>(true, true),
            false,
            true,
            originalMappingParameters
        );
        luceneFieldMapper.parseCreateField(parseContext, TEST_DIMENSION, VectorDataType.FLOAT);

        // Document should have 2 fields: one for VectorField (binary doc values) and one for KnnFloatVectorField
        List<IndexableField> fields = document.getFields();
        assertEquals(2, fields.size());
        IndexableField field1 = fields.get(0);
        IndexableField field2 = fields.get(1);

        VectorField vectorField;
        KnnFloatVectorField knnVectorField;
        if (field1 instanceof VectorField) {
            assertTrue(field2 instanceof KnnFloatVectorField);
            vectorField = (VectorField) field1;
            knnVectorField = (KnnFloatVectorField) field2;
        } else {
            assertTrue(field1 instanceof KnnFloatVectorField);
            assertTrue(field2 instanceof VectorField);
            knnVectorField = (KnnFloatVectorField) field1;
            vectorField = (VectorField) field2;
        }

        assertEquals(TEST_VECTOR_BYTES_REF, vectorField.binaryValue());
        assertEquals(VectorEncoding.FLOAT32, vectorField.fieldType().vectorEncoding());
        assertArrayEquals(TEST_VECTOR, knnVectorField.vectorValue(), 0.001f);

        // Test when doc values are disabled
        document = new ParseContext.Document();
        contentPath = new ContentPath();
        parseContext = mock(ParseContext.class);
        when(parseContext.doc()).thenReturn(document);
        when(parseContext.path()).thenReturn(contentPath);
        when(parseContext.parser()).thenReturn(createXContentParser(VectorDataType.FLOAT));
        when(parseContext.indexSettings()).thenReturn(indexSettingsMock);

        knnMethodConfigContext = KNNMethodConfigContext.builder()
            .vectorDataType(VectorDataType.FLOAT)
            .versionCreated(CURRENT)
            .dimension(TEST_DIMENSION)
            .build();
        originalMappingParameters = new OriginalMappingParameters(
            VectorDataType.FLOAT,
            TEST_DIMENSION,
            luceneMethodContext,
            Mode.NOT_CONFIGURED.getName(),
            CompressionLevel.NOT_CONFIGURED.getName(),
            null,
            SpaceType.UNDEFINED.getValue(),
            KNNEngine.UNDEFINED.getName()
        );
        originalMappingParameters.setResolvedKnnMethodContext(originalMappingParameters.getKnnMethodContext());
        luceneFieldMapper = EngineFieldMapper.createFieldMapper(
            TEST_FIELD_NAME,
            TEST_FIELD_NAME,
            Collections.emptyMap(),
            knnMethodConfigContext,
            FieldMapper.MultiFields.empty(),
            FieldMapper.CopyTo.empty(),
            new Explicit<>(true, true),
            false,
            false,
            originalMappingParameters
        );
        luceneFieldMapper.parseCreateField(parseContext, TEST_DIMENSION, VectorDataType.FLOAT);

        // Document should have 1 field: one for KnnVectorField
        fields = document.getFields();
        assertEquals(1, fields.size());
        IndexableField field = fields.get(0);
        assertTrue(field instanceof KnnFloatVectorField);
        knnVectorField = (KnnFloatVectorField) field;
        assertArrayEquals(TEST_VECTOR, knnVectorField.vectorValue(), 0.001f);
    }

    @SneakyThrows
    public void testLuceneFieldMapper_parseCreateField_docValues_withBytes() {
        IndexSettings indexSettingsMock = mock(IndexSettings.class);
        when(indexSettingsMock.getSettings()).thenReturn(Settings.EMPTY);
        ParseContext.Document document = new ParseContext.Document();
        ContentPath contentPath = new ContentPath();
        ParseContext parseContext = mock(ParseContext.class);
        when(parseContext.doc()).thenReturn(document);
        when(parseContext.path()).thenReturn(contentPath);
        when(parseContext.indexSettings()).thenReturn(indexSettingsMock);

        MethodComponentContext methodComponentContext = new MethodComponentContext(METHOD_HNSW, Collections.emptyMap());
        KNNMethodContext luceneByteKnnMethodContext = new KNNMethodContext(KNNEngine.LUCENE, SpaceType.DEFAULT, methodComponentContext);

        OriginalMappingParameters originalMappingParameters = new OriginalMappingParameters(
            VectorDataType.BYTE,
            TEST_DIMENSION,
            luceneByteKnnMethodContext,
            Mode.NOT_CONFIGURED.getName(),
            CompressionLevel.NOT_CONFIGURED.getName(),
            null,
            SpaceType.UNDEFINED.getValue(),
            KNNEngine.UNDEFINED.getName()
        );
        originalMappingParameters.setResolvedKnnMethodContext(originalMappingParameters.getKnnMethodContext());

        EngineFieldMapper luceneFieldMapper = Mockito.spy(
            EngineFieldMapper.createFieldMapper(
                TEST_FIELD_NAME,
                TEST_FIELD_NAME,
                Collections.emptyMap(),
                KNNMethodConfigContext.builder()
                    .vectorDataType(VectorDataType.BYTE)
                    .versionCreated(CURRENT)
                    .dimension(TEST_DIMENSION)
                    .build(),
                FieldMapper.MultiFields.empty(),
                FieldMapper.CopyTo.empty(),
                new Explicit<>(true, true),
                false,
                true,
                originalMappingParameters
            )
        );
        doReturn(Optional.of(TEST_BYTE_VECTOR)).when(luceneFieldMapper)
            .getBytesFromContext(parseContext, TEST_DIMENSION, VectorDataType.BYTE);
        doNothing().when(luceneFieldMapper).validatePreparse();

        luceneFieldMapper.parseCreateField(parseContext, TEST_DIMENSION, VectorDataType.BYTE);

        // Document should have 2 fields: one for VectorField (binary doc values) and one for KnnByteVectorField
        List<IndexableField> fields = document.getFields();
        assertEquals(2, fields.size());
        IndexableField field1 = fields.get(0);
        IndexableField field2 = fields.get(1);

        VectorField vectorField;
        KnnByteVectorField knnByteVectorField;
        if (field1 instanceof VectorField) {
            assertTrue(field2 instanceof KnnByteVectorField);
            vectorField = (VectorField) field1;
            knnByteVectorField = (KnnByteVectorField) field2;
        } else {
            assertTrue(field1 instanceof KnnByteVectorField);
            assertTrue(field2 instanceof VectorField);
            knnByteVectorField = (KnnByteVectorField) field1;
            vectorField = (VectorField) field2;
        }

        assertEquals(TEST_BYTE_VECTOR_BYTES_REF, vectorField.binaryValue());
        assertArrayEquals(TEST_BYTE_VECTOR, knnByteVectorField.vectorValue());

        // Test when doc values are disabled
        document = new ParseContext.Document();
        contentPath = new ContentPath();
        parseContext = mock(ParseContext.class);
        when(parseContext.doc()).thenReturn(document);
        when(parseContext.path()).thenReturn(contentPath);
        when(parseContext.indexSettings()).thenReturn(indexSettingsMock);

        luceneFieldMapper = Mockito.spy(
            EngineFieldMapper.createFieldMapper(
                TEST_FIELD_NAME,
                TEST_FIELD_NAME,
                Collections.emptyMap(),
                KNNMethodConfigContext.builder()
                    .vectorDataType(VectorDataType.BYTE)
                    .versionCreated(CURRENT)
                    .dimension(TEST_DIMENSION)
                    .build(),
                FieldMapper.MultiFields.empty(),
                FieldMapper.CopyTo.empty(),
                new Explicit<>(true, true),
                false,
                false,
                originalMappingParameters
            )
        );
        doReturn(Optional.of(TEST_BYTE_VECTOR)).when(luceneFieldMapper)
            .getBytesFromContext(parseContext, TEST_DIMENSION, VectorDataType.BYTE);
        doNothing().when(luceneFieldMapper).validatePreparse();

        luceneFieldMapper.parseCreateField(parseContext, TEST_DIMENSION, VectorDataType.BYTE);

        // Document should have 1 field: one for KnnByteVectorField
        fields = document.getFields();
        assertEquals(1, fields.size());
        IndexableField field = fields.get(0);
        assertTrue(field instanceof KnnByteVectorField);
        knnByteVectorField = (KnnByteVectorField) field;
        assertArrayEquals(TEST_BYTE_VECTOR, knnByteVectorField.vectorValue());
    }

    public void testTypeParser_whenBinaryFaissHNSW_thenValid() throws IOException {
        testTypeParserWithBinaryDataType(KNNEngine.FAISS, SpaceType.HAMMING, METHOD_HNSW, 8, null);
    }

    public void testTypeParser_whenBinaryWithInvalidDimension_thenException() throws IOException {
        testTypeParserWithBinaryDataType(KNNEngine.FAISS, SpaceType.HAMMING, METHOD_HNSW, 4, "should be multiply of 8");
    }

    public void testTypeParser_whenBinaryFaissHNSWWithInvalidSpaceType_thenException() throws IOException {
        for (SpaceType spaceType : SpaceType.values()) {
            if (SpaceType.UNDEFINED == spaceType || SpaceType.HAMMING == spaceType) {
                continue;
            }
            testTypeParserWithBinaryDataType(KNNEngine.FAISS, spaceType, METHOD_HNSW, 8, "is not supported with");
        }
    }

    private void testTypeParserWithBinaryDataType(
        KNNEngine knnEngine,
        SpaceType spaceType,
        String method,
        int dimension,
        String expectedErrMsg
    ) throws IOException {
        // Check legacy is picked up if model context and method context are not set
        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);
        String fieldName = "test-field-name-1";
        String indexName = "test-index";

        // Setup settings
        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .field(VECTOR_DATA_TYPE_FIELD, VectorDataType.BINARY.getValue())
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, knnEngine.getName())
            .endObject()
            .endObject();

        if (expectedErrMsg == null) {
            KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
                fieldName,
                xContentBuilderToMap(xContentBuilder),
                buildParserContext(indexName, settings)
            );

            assertEquals(spaceType, builder.getOriginalParameters().getResolvedKnnMethodContext().getSpaceType());
        } else {
            Exception ex = expectThrows(Exception.class, () -> {
                typeParser.parse(fieldName, xContentBuilderToMap(xContentBuilder), buildParserContext(indexName, settings));
            });
            assertTrue(ex.getMessage(), ex.getMessage().contains(expectedErrMsg));
        }
    }

    public void testTypeParser_whenBinaryFaissHNSWWithSQ_thenException() throws IOException {
        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);
        // Setup settings
        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, 8)
            .field(VECTOR_DATA_TYPE_FIELD, VectorDataType.BINARY.getValue())
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .startObject(PARAMETERS)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQ)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Exception ex = expectThrows(
            Exception.class,
            () -> typeParser.parse("test", xContentBuilderToMap(xContentBuilder), buildParserContext("test", settings))
        );
        assertTrue(ex.getMessage(), ex.getMessage().contains("parameter validation failed for MethodComponentContext parameter [encoder]"));
    }

    public void testBuilder_whenBinaryWithLegacyKNNDisabled_thenValid() {
        // Check legacy is picked up if model context and method context are not set
        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.Builder builder = new KNNVectorFieldMapper.Builder("test-field-name-1", modelDao, CURRENT, null, null);
        builder.vectorDataType.setValue(VectorDataType.BINARY);
        builder.dimension.setValue(8);

        // Setup settings
        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, false).build();

        builder.setOriginalParameters(new OriginalMappingParameters(builder));
        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper instanceof FlatVectorFieldMapper);
    }

    public void testTypeParser_whenBinaryWithLegacyKNNEnabled_thenValid() throws IOException {
        // Check legacy is picked up if model context and method context are not set
        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);
        String fieldName = "test-field-name-1";
        String indexName = "test-index";

        // Setup settings
        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, 8)
            .field(VECTOR_DATA_TYPE_FIELD, VectorDataType.BINARY.getValue())
            .endObject();

        final KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(xContentBuilder),
            buildParserContext(indexName, settings)
        );
        assertEquals(SpaceType.HAMMING, builder.getOriginalParameters().getResolvedKnnMethodContext().getSpaceType());
    }

    public void testBuild_whenInvalidCharsInFieldName_thenThrowException() {
        for (char disallowChar : Strings.INVALID_FILENAME_CHARS) {
            // When an invalid vector field name was given.
            final String invalidVectorFieldName = "fieldname" + disallowChar;

            // Prepare context.
            Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();
            Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());

            // IllegalArgumentException should be thrown.
            Exception e = assertThrows(IllegalArgumentException.class, () -> {
                new KNNVectorFieldMapper.Builder(invalidVectorFieldName, null, CURRENT, null, null).build(builderContext);
            });
            assertTrue(e.getMessage(), e.getMessage().contains("Vector field name must not include"));
        }
    }

    public void testTypeParser_whenModeAndCompressionAreSet_thenHandle() throws IOException {
        int dimension = 16;
        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();
        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);

        // Default to faiss and ensure legacy is in use
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .endObject();
        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            TEST_FIELD_NAME,
            xContentBuilderToMap(xContentBuilder),
            buildParserContext(TEST_INDEX_NAME, settings)
        );
        assertNull(builder.getOriginalParameters().getKnnMethodContext());
        assertTrue(builder.getOriginalParameters().isLegacyMapping());
        validateBuilderAfterParsing(
            builder,
            KNNEngine.DEFAULT,
            SpaceType.L2,
            VectorDataType.FLOAT,
            CompressionLevel.x1,
            CompressionLevel.NOT_CONFIGURED,
            Mode.NOT_CONFIGURED,
            false
        );

        // If mode is in memory and 1x compression, again use default legacy
        xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x1.getName())
            .field(MODE_PARAMETER, Mode.IN_MEMORY.getName())
            .endObject();
        builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            TEST_FIELD_NAME,
            xContentBuilderToMap(xContentBuilder),
            buildParserContext(TEST_INDEX_NAME, settings)
        );
        assertNull(builder.getOriginalParameters().getKnnMethodContext());
        assertFalse(builder.getOriginalParameters().isLegacyMapping());
        validateBuilderAfterParsing(
            builder,
            KNNEngine.FAISS,
            SpaceType.L2,
            VectorDataType.FLOAT,
            CompressionLevel.x1,
            CompressionLevel.x1,
            Mode.IN_MEMORY,
            false
        );

        // Default on disk is faiss with 32x binary quant
        xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .field(MODE_PARAMETER, Mode.ON_DISK.getName())
            .endObject();

        builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            TEST_FIELD_NAME,
            xContentBuilderToMap(xContentBuilder),
            buildParserContext(TEST_INDEX_NAME, settings)
        );
        validateBuilderAfterParsing(
            builder,
            KNNEngine.FAISS,
            SpaceType.L2,
            VectorDataType.FLOAT,
            CompressionLevel.x32,
            CompressionLevel.NOT_CONFIGURED,
            Mode.ON_DISK,
            true
        );

        // Ensure 2x does not use binary quantization
        xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x2.getName())
            .endObject();

        builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            TEST_FIELD_NAME,
            xContentBuilderToMap(xContentBuilder),
            buildParserContext(TEST_INDEX_NAME, settings)
        );
        validateBuilderAfterParsing(
            builder,
            KNNEngine.FAISS,
            SpaceType.L2,
            VectorDataType.FLOAT,
            CompressionLevel.x2,
            CompressionLevel.x2,
            Mode.NOT_CONFIGURED,
            false
        );

        // For 8x ensure that it does use binary quantization
        xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .field(MODE_PARAMETER, Mode.ON_DISK.getName())
            .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x8.getName())
            .endObject();

        builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            TEST_FIELD_NAME,
            xContentBuilderToMap(xContentBuilder),
            buildParserContext(TEST_INDEX_NAME, settings)
        );
        validateBuilderAfterParsing(
            builder,
            KNNEngine.FAISS,
            SpaceType.L2,
            VectorDataType.FLOAT,
            CompressionLevel.x8,
            CompressionLevel.x8,
            Mode.ON_DISK,
            true
        );

        // For 4x compression on disk, use Lucene
        xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .field(MODE_PARAMETER, Mode.ON_DISK.getName())
            .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x4.getName())
            .endObject();

        builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            TEST_FIELD_NAME,
            xContentBuilderToMap(xContentBuilder),
            buildParserContext(TEST_INDEX_NAME, settings)
        );
        validateBuilderAfterParsing(
            builder,
            KNNEngine.LUCENE,
            SpaceType.L2,
            VectorDataType.FLOAT,
            CompressionLevel.x4,
            CompressionLevel.x4,
            Mode.ON_DISK,
            false
        );

        // For 4x compression in memory, use Lucene
        xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .field(MODE_PARAMETER, Mode.IN_MEMORY.getName())
            .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x4.getName())
            .endObject();

        builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            TEST_FIELD_NAME,
            xContentBuilderToMap(xContentBuilder),
            buildParserContext(TEST_INDEX_NAME, settings)
        );
        validateBuilderAfterParsing(
            builder,
            KNNEngine.LUCENE,
            SpaceType.L2,
            VectorDataType.FLOAT,
            CompressionLevel.x4,
            CompressionLevel.x4,
            Mode.IN_MEMORY,
            false
        );

        // For override, ensure compression is correct
        xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, KNNEngine.FAISS)
            .startObject(PARAMETERS)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, QFrameBitEncoder.NAME)
            .startObject(PARAMETERS)
            .field(QFrameBitEncoder.BITCOUNT_PARAM, CompressionLevel.x16.numBitsForFloat32())
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            TEST_FIELD_NAME,
            xContentBuilderToMap(xContentBuilder),
            buildParserContext(TEST_INDEX_NAME, settings)
        );
        validateBuilderAfterParsing(
            builder,
            KNNEngine.FAISS,
            SpaceType.L2,
            VectorDataType.FLOAT,
            CompressionLevel.x16,
            CompressionLevel.NOT_CONFIGURED,
            Mode.NOT_CONFIGURED,
            true
        );

        // Override with conflicting compression levels should fail
        XContentBuilder invalidXContentBuilder1 = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x4.getName())
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, KNNEngine.FAISS)
            .startObject(PARAMETERS)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, QFrameBitEncoder.NAME)
            .startObject(PARAMETERS)
            .field(QFrameBitEncoder.BITCOUNT_PARAM, CompressionLevel.x16.numBitsForFloat32())
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        expectThrows(
            ValidationException.class,
            () -> typeParser.parse(
                TEST_FIELD_NAME,
                xContentBuilderToMap(invalidXContentBuilder1),
                buildParserContext(TEST_INDEX_NAME, settings)
            )
        );

        // Invalid if vector data type is binary
        XContentBuilder invalidXContentBuilder2 = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .field(VECTOR_DATA_TYPE_FIELD, VectorDataType.BINARY.getValue())
            .field(MODE_PARAMETER, Mode.IN_MEMORY.getName())
            .endObject();

        expectThrows(
            MapperParsingException.class,
            () -> typeParser.parse(
                TEST_FIELD_NAME,
                xContentBuilderToMap(invalidXContentBuilder2),
                buildParserContext(TEST_INDEX_NAME, settings)
            )
        );

        // Invalid if engine doesnt support the compression
        XContentBuilder invalidXContentBuilder3 = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x4.getName())
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, KNNEngine.FAISS)
            .endObject()
            .endObject();

        expectThrows(
            ValidationException.class,
            () -> typeParser.parse(
                TEST_FIELD_NAME,
                xContentBuilderToMap(invalidXContentBuilder3),
                buildParserContext(TEST_INDEX_NAME, settings)
            )
        );
    }

    public void testKNNVectorFieldMapper_UpdateMethodParameter_ThrowsException() throws IOException {
        String fieldName = "test-field-name";
        String indexName = "test-index-name";

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();
        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);

        // Define initial mapping with method parameter
        XContentBuilder initialMapping = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, 4)
            .startObject(KNN_METHOD)
            .field("engine", "faiss")
            .field("name", "hnsw")
            .endObject()
            .endObject();

        KNNVectorFieldMapper.Builder initialBuilder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(initialMapping),
            buildParserContext(indexName, settings)
        );

        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper initialMapper = initialBuilder.build(builderContext);

        // Define updated mapping without the method parameter
        XContentBuilder updatedMapping = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, 4)
            .endObject();

        KNNVectorFieldMapper.Builder updatedBuilder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(updatedMapping),
            buildParserContext(indexName, settings)
        );

        KNNVectorFieldMapper updatedMapper = updatedBuilder.build(builderContext);

        // Expect an IllegalArgumentException when merging the updated mapping
        IllegalArgumentException exception = expectThrows(IllegalArgumentException.class, () -> initialMapper.merge(updatedMapper));
        assertTrue(exception.getMessage().contains("Cannot update parameter [method] from [hnsw] to [null]"));
    }

    public void testKNNVectorFieldMapper_UpdateDimensionParameter_Succeeds() throws IOException {
        String fieldName = TEST_FIELD_NAME;
        String indexName = TEST_INDEX_NAME;

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();
        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);

        // Define updated mapping with the same method parameter
        XContentBuilder updatedMapping = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, 8)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, "faiss")
            .endObject()
            .endObject();

        KNNVectorFieldMapper.Builder updatedBuilder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(updatedMapping),
            buildParserContext(indexName, settings)
        );

        assertEquals(8, updatedBuilder.getOriginalParameters().getDimension());
    }

    public void testKNNVectorFieldMapper_PartialUpdateMethodParameter_ThrowsException() throws IOException {
        String fieldName = TEST_FIELD_NAME;
        String indexName = TEST_INDEX_NAME;

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();
        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> modelDao);

        XContentBuilder updatedMapping = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, 4)
            .startObject(KNN_METHOD)
            .field(NAME, "")
            .field(KNN_ENGINE, "faiss")
            .endObject()
            .endObject();

        MapperParsingException exception = expectThrows(
            MapperParsingException.class,
            () -> typeParser.parse(fieldName, xContentBuilderToMap(updatedMapping), buildParserContext(indexName, settings))
        );

        assertTrue(exception.getMessage().contains("name needs to be set"));
    }

    private void validateBuilderAfterParsing(
        KNNVectorFieldMapper.Builder builder,
        KNNEngine expectedEngine,
        SpaceType expectedSpaceType,
        VectorDataType expectedVectorDataType,
        CompressionLevel expectedResolvedCompressionLevel,
        CompressionLevel expectedOriginalCompressionLevel,
        Mode expectedMode,
        boolean shouldUsesBinaryQFramework
    ) {
        assertEquals(expectedEngine, builder.getOriginalParameters().getResolvedKnnMethodContext().getKnnEngine());
        assertEquals(expectedSpaceType, builder.getOriginalParameters().getResolvedKnnMethodContext().getSpaceType());
        assertEquals(expectedVectorDataType, builder.getKnnMethodConfigContext().getVectorDataType());

        assertEquals(expectedResolvedCompressionLevel, builder.getKnnMethodConfigContext().getCompressionLevel());
        assertEquals(expectedOriginalCompressionLevel, CompressionLevel.fromName(builder.getOriginalParameters().getCompressionLevel()));
        assertEquals(expectedMode, Mode.fromName(builder.getOriginalParameters().getMode()));
        assertEquals(expectedMode, builder.getKnnMethodConfigContext().getMode());
        assertFalse(builder.getOriginalParameters().getResolvedKnnMethodContext().getMethodComponentContext().getParameters().isEmpty());

        if (shouldUsesBinaryQFramework) {
            assertEquals(
                QFrameBitEncoder.NAME,
                ((MethodComponentContext) builder.getOriginalParameters()
                    .getResolvedKnnMethodContext()
                    .getMethodComponentContext()
                    .getParameters()
                    .get(METHOD_ENCODER_PARAMETER)).getName()
            );
            assertEquals(
                expectedResolvedCompressionLevel.numBitsForFloat32(),
                (int) ((MethodComponentContext) builder.getOriginalParameters()
                    .getResolvedKnnMethodContext()
                    .getMethodComponentContext()
                    .getParameters()
                    .get(METHOD_ENCODER_PARAMETER)).getParameters().get(QFrameBitEncoder.BITCOUNT_PARAM)
            );
        } else {
            assertTrue(
                builder.getOriginalParameters().getResolvedKnnMethodContext().getMethodComponentContext().getParameters().isEmpty()
                    || builder.getOriginalParameters()
                        .getResolvedKnnMethodContext()
                        .getMethodComponentContext()
                        .getParameters()
                        .containsKey(METHOD_ENCODER_PARAMETER) == false
                    || QFrameBitEncoder.NAME.equals(
                        ((MethodComponentContext) builder.getOriginalParameters()
                            .getResolvedKnnMethodContext()
                            .getMethodComponentContext()
                            .getParameters()
                            .get(METHOD_ENCODER_PARAMETER)).getName()
                    ) == false
            );
        }
    }

    private XContentBuilder createXContentForFieldMapping(
        SpaceType topLevelSpaceType,
        SpaceType methodSpaceType,
        KNNEngine knnEngine,
        VectorDataType vectorDataType,
        int dimension
    ) throws IOException {
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension);

        if (topLevelSpaceType != null && topLevelSpaceType != SpaceType.UNDEFINED) {
            xContentBuilder.field(KNNConstants.TOP_LEVEL_PARAMETER_SPACE_TYPE, topLevelSpaceType.getValue());
        }
        if (vectorDataType != null) {
            xContentBuilder.field(VECTOR_DATA_TYPE_FIELD, vectorDataType.getValue());
        }
        xContentBuilder.startObject(KNN_METHOD).field(NAME, METHOD_HNSW);
        if (knnEngine != null) {
            xContentBuilder.field(KNN_ENGINE, knnEngine.getName());
        }
        if (methodSpaceType != null && methodSpaceType != SpaceType.UNDEFINED) {
            xContentBuilder.field(METHOD_PARAMETER_SPACE_TYPE, methodSpaceType.getValue());
        }
        xContentBuilder.endObject().endObject();
        return xContentBuilder;
    }

    private XContentBuilder createXContentForFieldMapping_Engine(
        KNNEngine methodKNNEngine,
        KNNEngine topLevelKNNEngine,
        SpaceType spaceType,
        VectorDataType vectorDataType,
        int dimension
    ) throws IOException {
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension);

        if (topLevelKNNEngine != null) {
            xContentBuilder.field(TOP_LEVEL_PARAMETER_ENGINE, topLevelKNNEngine.getName());
        }
        if (vectorDataType != null) {
            xContentBuilder.field(VECTOR_DATA_TYPE_FIELD, vectorDataType.getValue());
        }
        if (methodKNNEngine != null) {
            xContentBuilder.startObject(KNN_METHOD).field(NAME, METHOD_HNSW);
            if (spaceType != null && spaceType != SpaceType.UNDEFINED) {
                xContentBuilder.field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue());
            }
            xContentBuilder.field(KNN_ENGINE, methodKNNEngine.getName());
            xContentBuilder.endObject();
        }
        xContentBuilder.endObject();
        return xContentBuilder;
    }

    private static float[] createInitializedFloatArray(int dimension, float value) {
        float[] array = new float[dimension];
        Arrays.fill(array, value);
        return array;
    }

    private static byte[] createInitializedByteArray(int dimension, byte value) {
        byte[] array = new byte[dimension];
        Arrays.fill(array, value);
        return array;
    }

    private XContentParser createXContentParser(final VectorDataType dataType) throws IOException {
        final String vectorString;
        if (dataType == VectorDataType.FLOAT) {
            vectorString = Arrays.toString(TEST_VECTOR);
        } else {
            vectorString = Arrays.toString(TEST_BYTE_VECTOR);
        }

        XContentParser parser = XContentHelper.createParser(
            NamedXContentRegistry.EMPTY,
            LoggingDeprecationHandler.INSTANCE,
            new BytesArray(new BytesArray("{\"" + TEST_FIELD_NAME + "\":" + vectorString + "}").toBytesRef()),
            MediaTypeRegistry.JSON
        );
        // We need to move to 3rd token, as at start XContentParser is at null, first nextToken call will move the
        // parser to { aka START_OBJECT, the next call will move it to FIELD_NAME and after that it will move to [
        // aka START_ARRAY which is what we get when we parse the vectors.
        parser.nextToken();
        parser.nextToken();
        parser.nextToken();
        return parser;
    }

    public IndexMetadata buildIndexMetaData(String indexName, Settings settings) {
        return IndexMetadata.builder(indexName)
            .settings(settings)
            .numberOfShards(1)
            .numberOfReplicas(0)
            .version(7)
            .mappingVersion(0)
            .settingsVersion(0)
            .aliasesVersion(0)
            .creationDate(0)
            .build();
    }

    public Mapper.TypeParser.ParserContext buildParserContext(String indexName, Settings settings) {
        return dobuildParserContext(indexName, settings, CURRENT);
    }

    public Mapper.TypeParser.ParserContext buildLegacyParserContext(String indexName, Settings settings, Version version) {
        return dobuildParserContext(indexName, settings, version);
    }

    public Mapper.TypeParser.ParserContext dobuildParserContext(String indexName, Settings settings, Version version) {
        IndexSettings indexSettings = new IndexSettings(
            buildIndexMetaData(indexName, settings),
            Settings.EMPTY,
            new IndexScopedSettings(Settings.EMPTY, new HashSet<>(IndexScopedSettings.BUILT_IN_INDEX_SETTINGS))
        );
        MapperService mapperService = mock(MapperService.class);
        when(mapperService.getIndexSettings()).thenReturn(indexSettings);

        // Setup blank
        ModelDao mockModelDao = mock(ModelDao.class);
        return new Mapper.TypeParser.ParserContext(
            null,
            mapperService,
            type -> new KNNVectorFieldMapper.TypeParser(() -> mockModelDao),
            version,
            null,
            null,
            null
        );
    }
}

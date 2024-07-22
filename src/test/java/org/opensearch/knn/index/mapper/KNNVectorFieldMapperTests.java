/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import com.google.common.collect.ImmutableMap;
import lombok.SneakyThrows;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnVectorField;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.util.BytesRef;
import org.mockito.Mockito;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.Explicit;
import org.opensearch.common.ValidationException;
import org.opensearch.common.settings.IndexScopedSettings;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.ContentPath;
import org.opensearch.index.mapper.FieldMapper;
import org.opensearch.index.mapper.Mapper;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.VectorField;
import org.opensearch.knn.index.codec.util.KNNVectorSerializerFactory;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;

import java.io.IOException;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.Version.CURRENT;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FP16_MAX_VALUE;
import static org.opensearch.knn.common.KNNConstants.FP16_MIN_VALUE;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.NMSLIB_NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;
import static org.opensearch.knn.index.VectorDataType.SUPPORTED_VECTOR_DATA_TYPES;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.clipVectorValueToFP16Range;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.validateFP16VectorValue;

public class KNNVectorFieldMapperTests extends KNNTestCase {

    private static final String TEST_FIELD_NAME = "test-field-name";

    private static final int TEST_DIMENSION = 17;

    private static final float TEST_VECTOR_VALUE = 1.5f;

    private static final float[] TEST_VECTOR = createInitializedFloatArray(TEST_DIMENSION, TEST_VECTOR_VALUE);

    private static final byte TEST_BYTE_VECTOR_VALUE = 10;
    private static final byte[] TEST_BYTE_VECTOR = createInitializedByteArray(TEST_DIMENSION, TEST_BYTE_VECTOR_VALUE);

    private static final BytesRef TEST_VECTOR_BYTES_REF = new BytesRef(
        KNNVectorSerializerFactory.getDefaultSerializer().floatToByteArray(TEST_VECTOR)
    );
    private static final BytesRef TEST_BYTE_VECTOR_BYTES_REF = new BytesRef(TEST_BYTE_VECTOR);
    private static final String DIMENSION_FIELD_NAME = "dimension";
    private static final String KNN_VECTOR_TYPE = "knn_vector";
    private static final String TYPE_FIELD_NAME = "type";

    public void testBuilder_getParameters() {
        String fieldName = "test-field-name";
        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.Builder builder = new KNNVectorFieldMapper.Builder(fieldName, modelDao, CURRENT);

        assertEquals(7, builder.getParameters().size());
        List<String> actualParams = builder.getParameters().stream().map(a -> a.name).collect(Collectors.toList());
        List<String> expectedParams = Arrays.asList("store", "doc_values", DIMENSION, VECTOR_DATA_TYPE_FIELD, "meta", KNN_METHOD, MODEL_ID);
        assertEquals(expectedParams, actualParams);
    }

    public void testBuilder_build_fromKnnMethodContext() {
        // Check that knnMethodContext takes precedent over both model and legacy
        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.Builder builder = new KNNVectorFieldMapper.Builder("test-field-name-1", modelDao, CURRENT);

        SpaceType spaceType = SpaceType.COSINESIMIL;
        int m = 17;
        int efConstruction = 17;

        // Setup settings
        Settings settings = Settings.builder()
            .put(settings(CURRENT).build())
            .put(KNNSettings.KNN_SPACE_TYPE, spaceType.getValue())
            .put(KNNSettings.KNN_ALGO_PARAM_M, m)
            .put(KNNSettings.KNN_ALGO_PARAM_EF_CONSTRUCTION, efConstruction)
            .build();

        builder.knnMethodContext.setValue(
            new KNNMethodContext(
                KNNEngine.DEFAULT,
                spaceType,
                new MethodComponentContext(
                    METHOD_HNSW,
                    ImmutableMap.of(METHOD_PARAMETER_M, m, METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction)
                )
            )
        );

        builder.modelId.setValue("Random modelId");

        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper instanceof MethodFieldMapper);
        assertNotNull(knnVectorFieldMapper.knnMethod);
        assertNull(knnVectorFieldMapper.modelId);
    }

    public void testBuilder_build_fromModel() {
        // Check that modelContext takes precedent over legacy
        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.Builder builder = new KNNVectorFieldMapper.Builder("test-field-name-1", modelDao, CURRENT);

        SpaceType spaceType = SpaceType.COSINESIMIL;
        int m = 17;
        int efConstruction = 17;

        // Setup settings
        Settings settings = Settings.builder()
            .put(settings(CURRENT).build())
            .put(KNNSettings.KNN_SPACE_TYPE, spaceType.getValue())
            .put(KNNSettings.KNN_ALGO_PARAM_M, m)
            .put(KNNSettings.KNN_ALGO_PARAM_EF_CONSTRUCTION, efConstruction)
            .build();

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
            VectorDataType.FLOAT
        );
        builder.modelId.setValue(modelId);
        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());

        when(modelDao.getMetadata(modelId)).thenReturn(mockedModelMetadata);
        KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper instanceof ModelFieldMapper);
        assertNotNull(knnVectorFieldMapper.modelId);
        assertNull(knnVectorFieldMapper.knnMethod);
    }

    public void testBuilder_build_fromLegacy() {
        // Check legacy is picked up if model context and method context are not set
        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.Builder builder = new KNNVectorFieldMapper.Builder("test-field-name-1", modelDao, CURRENT);

        int m = 17;
        int efConstruction = 17;

        // Setup settings
        Settings settings = Settings.builder()
            .put(settings(CURRENT).build())
            .put(KNNSettings.KNN_ALGO_PARAM_M, m)
            .put(KNNSettings.KNN_ALGO_PARAM_EF_CONSTRUCTION, efConstruction)
            .build();

        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper instanceof LegacyFieldMapper);
        assertNull(knnVectorFieldMapper.modelId);
        assertNull(knnVectorFieldMapper.knnMethod);
        assertEquals(SpaceType.L2.getValue(), ((LegacyFieldMapper) knnVectorFieldMapper).spaceType);
    }

    public void testBuilder_whenKnnFalseWithBinary_thenSetHammingAsDefault() {
        // Check legacy is picked up if model context and method context are not set
        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.Builder builder = new KNNVectorFieldMapper.Builder("test-field-name-1", modelDao, CURRENT);
        builder.vectorDataType.setValue(VectorDataType.BINARY);
        builder.dimension.setValue(8);

        // Setup settings
        Settings settings = Settings.builder().put(settings(CURRENT).build()).build();

        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper instanceof LegacyFieldMapper);
        assertEquals(SpaceType.HAMMING.getValue(), ((LegacyFieldMapper) knnVectorFieldMapper).spaceType);
    }

    public void testBuilder_parse_fromKnnMethodContext_luceneEngine() throws IOException {
        String fieldName = "test-field-name";
        String indexName = "test-index-name";

        Settings settings = Settings.builder().put(settings(CURRENT).build()).build();

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

        Settings settings = Settings.builder().put(settings(CURRENT).build()).build();

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
        KNNVectorFieldMapper.Builder builderOverMaxDimension = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(xContentBuilderOverMaxDimension),
            buildParserContext(indexName, settings)
        );

        IllegalArgumentException ex = expectThrows(
            IllegalArgumentException.class,
            () -> builderOverMaxDimension.build(new Mapper.BuilderContext(settings, new ContentPath()))
        );
        assertEquals("Dimension value cannot be greater than 16000 for vector: test-field-name", ex.getMessage());

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

        Settings settings = Settings.builder().put(settings(CURRENT).build()).build();

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

        Settings settings = Settings.builder().put(settings(CURRENT).build()).build();

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

        Settings settings = Settings.builder().put(settings(CURRENT).build()).build();

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
    }

    public void testTypeParser_parse_fromLegacy() throws IOException {
        // Check that the particular values are set in builder
        String fieldName = "test-field-name";
        String indexName = "test-index-name";

        int m = 144;
        int efConstruction = 123;
        SpaceType spaceType = SpaceType.L2;
        Settings settings = Settings.builder()
            .put(settings(CURRENT).build())
            .put(KNNSettings.KNN_SPACE_TYPE, spaceType.getValue())
            .put(KNNSettings.KNN_ALGO_PARAM_M, m)
            .put(KNNSettings.KNN_ALGO_PARAM_EF_CONSTRUCTION, efConstruction)
            .build();

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

    public void testKNNVectorFieldMapper_merge_fromKnnMethodContext() throws IOException {
        String fieldName = "test-field-name";
        String indexName = "test-index-name";

        Settings settings = Settings.builder().put(settings(CURRENT).build()).build();

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
        assertEquals(knnVectorFieldMapper1.knnMethod, knnVectorFieldMapperMerge1.knnMethod);

        // merge with another mapper of the same field with same context
        KNNVectorFieldMapper knnVectorFieldMapper2 = builder.build(builderContext);
        KNNVectorFieldMapper knnVectorFieldMapperMerge2 = (KNNVectorFieldMapper) knnVectorFieldMapper1.merge(knnVectorFieldMapper2);
        assertEquals(knnVectorFieldMapper1.knnMethod, knnVectorFieldMapperMerge2.knnMethod);

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

        Settings settings = Settings.builder().put(settings(CURRENT).build()).build();

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
            VectorDataType.FLOAT
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
        assertEquals(knnVectorFieldMapper1.modelId, knnVectorFieldMapperMerge1.modelId);

        // merge with another mapper of the same field with same context
        KNNVectorFieldMapper knnVectorFieldMapper2 = builder.build(builderContext);
        KNNVectorFieldMapper knnVectorFieldMapperMerge2 = (KNNVectorFieldMapper) knnVectorFieldMapper1.merge(knnVectorFieldMapper2);
        assertEquals(knnVectorFieldMapper1.modelId, knnVectorFieldMapperMerge2.modelId);

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
    public void testLuceneFieldMapper_parseCreateField_docValues_withFloats() {
        // Create a lucene field mapper that creates a binary doc values field as well as KnnVectorField
        LuceneFieldMapper.CreateLuceneFieldMapperInput.CreateLuceneFieldMapperInputBuilder inputBuilder =
            createLuceneFieldMapperInputBuilder(VectorDataType.FLOAT);

        ParseContext.Document document = new ParseContext.Document();
        ContentPath contentPath = new ContentPath();
        ParseContext parseContext = mock(ParseContext.class);
        when(parseContext.doc()).thenReturn(document);
        when(parseContext.path()).thenReturn(contentPath);

        LuceneFieldMapper luceneFieldMapper = Mockito.spy(new LuceneFieldMapper(inputBuilder.build()));
        doReturn(Optional.of(TEST_VECTOR)).when(luceneFieldMapper)
            .getFloatsFromContext(parseContext, TEST_DIMENSION, new MethodComponentContext(METHOD_HNSW, Collections.emptyMap()));
        doNothing().when(luceneFieldMapper).validateIfCircuitBreakerIsNotTriggered();
        doNothing().when(luceneFieldMapper).validateIfKNNPluginEnabled();
        luceneFieldMapper.parseCreateField(
            parseContext,
            TEST_DIMENSION,
            luceneFieldMapper.fieldType().spaceType,
            luceneFieldMapper.fieldType().knnMethodContext.getMethodComponentContext(),
            VectorDataType.FLOAT
        );

        // Document should have 2 fields: one for VectorField (binary doc values) and one for KnnVectorField
        List<IndexableField> fields = document.getFields();
        assertEquals(2, fields.size());
        IndexableField field1 = fields.get(0);
        IndexableField field2 = fields.get(1);

        VectorField vectorField;
        KnnVectorField knnVectorField;
        if (field1 instanceof VectorField) {
            assertTrue(field2 instanceof KnnVectorField);
            vectorField = (VectorField) field1;
            knnVectorField = (KnnVectorField) field2;
        } else {
            assertTrue(field1 instanceof KnnVectorField);
            assertTrue(field2 instanceof VectorField);
            knnVectorField = (KnnVectorField) field1;
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

        inputBuilder.hasDocValues(false);
        luceneFieldMapper = Mockito.spy(new LuceneFieldMapper(inputBuilder.build()));
        doReturn(Optional.of(TEST_VECTOR)).when(luceneFieldMapper)
            .getFloatsFromContext(parseContext, TEST_DIMENSION, new MethodComponentContext(METHOD_HNSW, Collections.emptyMap()));
        doNothing().when(luceneFieldMapper).validateIfCircuitBreakerIsNotTriggered();
        doNothing().when(luceneFieldMapper).validateIfKNNPluginEnabled();

        luceneFieldMapper.parseCreateField(
            parseContext,
            TEST_DIMENSION,
            luceneFieldMapper.fieldType().spaceType,
            luceneFieldMapper.fieldType().knnMethodContext.getMethodComponentContext(),
            VectorDataType.FLOAT
        );

        // Document should have 1 field: one for KnnVectorField
        fields = document.getFields();
        assertEquals(1, fields.size());
        IndexableField field = fields.get(0);
        assertTrue(field instanceof KnnVectorField);
        knnVectorField = (KnnVectorField) field;
        assertArrayEquals(TEST_VECTOR, knnVectorField.vectorValue(), 0.001f);
    }

    @SneakyThrows
    public void testLuceneFieldMapper_parseCreateField_docValues_withBytes() {
        // Create a lucene field mapper that creates a binary doc values field as well as KnnByteVectorField

        LuceneFieldMapper.CreateLuceneFieldMapperInput.CreateLuceneFieldMapperInputBuilder inputBuilder =
            createLuceneFieldMapperInputBuilder(VectorDataType.BYTE);

        ParseContext.Document document = new ParseContext.Document();
        ContentPath contentPath = new ContentPath();
        ParseContext parseContext = mock(ParseContext.class);
        when(parseContext.doc()).thenReturn(document);
        when(parseContext.path()).thenReturn(contentPath);

        LuceneFieldMapper luceneFieldMapper = Mockito.spy(new LuceneFieldMapper(inputBuilder.build()));
        doReturn(Optional.of(TEST_BYTE_VECTOR)).when(luceneFieldMapper)
            .getBytesFromContext(parseContext, TEST_DIMENSION, VectorDataType.BYTE);
        doNothing().when(luceneFieldMapper).validateIfCircuitBreakerIsNotTriggered();
        doNothing().when(luceneFieldMapper).validateIfKNNPluginEnabled();

        luceneFieldMapper.parseCreateField(
            parseContext,
            TEST_DIMENSION,
            luceneFieldMapper.fieldType().spaceType,
            luceneFieldMapper.fieldType().knnMethodContext.getMethodComponentContext(),
            VectorDataType.BYTE
        );

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

        inputBuilder.hasDocValues(false);
        luceneFieldMapper = Mockito.spy(new LuceneFieldMapper(inputBuilder.build()));
        doReturn(Optional.of(TEST_BYTE_VECTOR)).when(luceneFieldMapper)
            .getBytesFromContext(parseContext, TEST_DIMENSION, VectorDataType.BYTE);
        doNothing().when(luceneFieldMapper).validateIfCircuitBreakerIsNotTriggered();
        doNothing().when(luceneFieldMapper).validateIfKNNPluginEnabled();

        luceneFieldMapper.parseCreateField(
            parseContext,
            TEST_DIMENSION,
            luceneFieldMapper.fieldType().spaceType,
            luceneFieldMapper.fieldType().knnMethodContext.getMethodComponentContext(),
            VectorDataType.BYTE
        );

        // Document should have 1 field: one for KnnByteVectorField
        fields = document.getFields();
        assertEquals(1, fields.size());
        IndexableField field = fields.get(0);
        assertTrue(field instanceof KnnByteVectorField);
        knnByteVectorField = (KnnByteVectorField) field;
        assertArrayEquals(TEST_BYTE_VECTOR, knnByteVectorField.vectorValue());
    }

    public void testValidateFp16VectorValue_outOfRange_throwsException() {
        IllegalArgumentException ex = expectThrows(IllegalArgumentException.class, () -> validateFP16VectorValue(65505.25f));
        assertTrue(
            ex.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "encoder name is set as [%s] and type is set as [%s] in index mapping. But, KNN vector values are not within in the FP16 range [%f, %f]",
                        ENCODER_SQ,
                        FAISS_SQ_ENCODER_FP16,
                        FP16_MIN_VALUE,
                        FP16_MAX_VALUE
                    )
                )
        );

        IllegalArgumentException ex1 = expectThrows(IllegalArgumentException.class, () -> validateFP16VectorValue(-65525.65f));
        assertTrue(
            ex1.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "encoder name is set as [%s] and type is set as [%s] in index mapping. But, KNN vector values are not within in the FP16 range [%f, %f]",
                        ENCODER_SQ,
                        FAISS_SQ_ENCODER_FP16,
                        FP16_MIN_VALUE,
                        FP16_MAX_VALUE
                    )
                )
        );
    }

    public void testClipVectorValuetoFP16Range_succeed() {
        assertEquals(65504.0f, clipVectorValueToFP16Range(65504.10f), 0.0f);
        assertEquals(65504.0f, clipVectorValueToFP16Range(1000000.89f), 0.0f);
        assertEquals(-65504.0f, clipVectorValueToFP16Range(-65504.10f), 0.0f);
        assertEquals(-65504.0f, clipVectorValueToFP16Range(-1000000.89f), 0.0f);
    }

    public void testBuilder_whenBinaryFaissHNSW_thenValid() {
        testBuilderWithBinaryDataType(KNNEngine.FAISS, SpaceType.UNDEFINED, METHOD_HNSW, 8, null);
    }

    public void testBuilder_whenBinaryWithInvalidDimension_thenException() {
        testBuilderWithBinaryDataType(KNNEngine.FAISS, SpaceType.UNDEFINED, METHOD_HNSW, 4, "should be multiply of 8");
    }

    public void testBuilder_whenBinaryFaissHNSWWithInvalidSpaceType_thenException() {
        for (SpaceType spaceType : SpaceType.values()) {
            if (SpaceType.UNDEFINED == spaceType || SpaceType.HAMMING == spaceType) {
                continue;
            }
            testBuilderWithBinaryDataType(KNNEngine.FAISS, spaceType, METHOD_HNSW, 8, "is not supported");
        }
    }

    public void testBuilder_whenBinaryNonFaiss_thenException() {
        testBuilderWithBinaryDataType(KNNEngine.LUCENE, SpaceType.UNDEFINED, METHOD_HNSW, 8, "is only supported for");
        testBuilderWithBinaryDataType(KNNEngine.NMSLIB, SpaceType.UNDEFINED, METHOD_HNSW, 8, "is only supported for");
    }

    private void testBuilderWithBinaryDataType(
        KNNEngine knnEngine,
        SpaceType spaceType,
        String method,
        int dimension,
        String expectedErrMsg
    ) {
        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.Builder builder = new KNNVectorFieldMapper.Builder("test-field-name-1", modelDao, CURRENT);

        // Setup settings
        Settings settings = Settings.builder().put(settings(CURRENT).build()).build();

        builder.knnMethodContext.setValue(
            new KNNMethodContext(knnEngine, spaceType, new MethodComponentContext(method, Collections.emptyMap()))
        );
        builder.vectorDataType.setValue(VectorDataType.BINARY);
        builder.dimension.setValue(dimension);

        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        if (expectedErrMsg == null) {
            KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);
            assertTrue(knnVectorFieldMapper instanceof MethodFieldMapper);
            if (SpaceType.UNDEFINED == spaceType) {
                assertEquals(SpaceType.HAMMING, knnVectorFieldMapper.fieldType().spaceType);
            }
        } else {
            Exception ex = expectThrows(Exception.class, () -> builder.build(builderContext));
            assertTrue(ex.getMessage(), ex.getMessage().contains(expectedErrMsg));
        }
    }

    public void testBuilder_whenBinaryFaissHNSWWithSQ_thenException() {
        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.Builder builder = new KNNVectorFieldMapper.Builder("test-field-name-1", modelDao, CURRENT);

        // Setup settings
        Settings settings = Settings.builder().put(settings(CURRENT).build()).build();

        builder.knnMethodContext.setValue(
            new KNNMethodContext(
                KNNEngine.FAISS,
                SpaceType.HAMMING,
                new MethodComponentContext(
                    METHOD_HNSW,
                    Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_SQ, Collections.emptyMap()))
                )
            )
        );
        builder.vectorDataType.setValue(VectorDataType.BINARY);
        builder.dimension.setValue(8);

        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        Exception ex = expectThrows(Exception.class, () -> builder.build(builderContext));
        assertTrue(ex.getMessage(), ex.getMessage().contains("data type does not support"));
    }

    public void testBuilder_whenBinaryWithLegacyKNNDisabled_thenValid() {
        // Check legacy is picked up if model context and method context are not set
        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.Builder builder = new KNNVectorFieldMapper.Builder("test-field-name-1", modelDao, CURRENT);
        builder.vectorDataType.setValue(VectorDataType.BINARY);
        builder.dimension.setValue(8);

        // Setup settings
        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, false).build();

        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper instanceof LegacyFieldMapper);
    }

    public void testBuilder_whenBinaryWithLegacyKNNEnabled_thenException() {
        // Check legacy is picked up if model context and method context are not set
        ModelDao modelDao = mock(ModelDao.class);
        KNNVectorFieldMapper.Builder builder = new KNNVectorFieldMapper.Builder("test-field-name-1", modelDao, CURRENT);
        builder.vectorDataType.setValue(VectorDataType.BINARY);
        builder.dimension.setValue(8);

        // Setup settings
        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        Exception ex = expectThrows(Exception.class, () -> builder.build(builderContext));
        assertTrue(ex.getMessage(), ex.getMessage().contains("is not supported for"));
    }

    private LuceneFieldMapper.CreateLuceneFieldMapperInput.CreateLuceneFieldMapperInputBuilder createLuceneFieldMapperInputBuilder(
        VectorDataType vectorDataType
    ) {
        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.DEFAULT,
            new MethodComponentContext(METHOD_HNSW, Collections.emptyMap())
        );

        KNNVectorFieldMapper.KNNVectorFieldType knnVectorFieldType = new KNNVectorFieldMapper.KNNVectorFieldType(
            TEST_FIELD_NAME,
            Collections.emptyMap(),
            TEST_DIMENSION,
            knnMethodContext,
            vectorDataType
        );

        return LuceneFieldMapper.CreateLuceneFieldMapperInput.builder()
            .name(TEST_FIELD_NAME)
            .mappedFieldType(knnVectorFieldType)
            .multiFields(FieldMapper.MultiFields.empty())
            .copyTo(FieldMapper.CopyTo.empty())
            .hasDocValues(true)
            .vectorDataType(vectorDataType)
            .ignoreMalformed(new Explicit<>(true, true))
            .knnMethodContext(knnMethodContext);
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
            CURRENT,
            null,
            null,
            null
        );

    }
}

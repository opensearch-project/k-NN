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
/*
 *   Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License").
 *   You may not use this file except in compliance with the License.
 *   A copy of the License is located at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   or in the "license" file accompanying this file. This file is distributed
 *   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *   express or implied. See the License for the specific language governing
 *   permissions and limitations under the License.
 */

package org.opensearch.knn.index;

import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.ValidationException;
import org.opensearch.common.settings.IndexScopedSettings;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.ContentPath;
import org.opensearch.index.mapper.Mapper;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;
import java.util.HashSet;

import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODEL;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.Version.CURRENT;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class KNNVectorFieldMapperTests extends KNNTestCase {

    public void testBuilder_getParameters() {
        String fieldName = "test-field-name";
        KNNVectorFieldMapper.Builder builder = new KNNVectorFieldMapper.Builder(fieldName);
        assertEquals(6, builder.getParameters().size());
    }

    public void testBuilder_build_fromKnnMethodContext() {
        // Check that knnMethodContext takes precedent over both model and legacy
        KNNVectorFieldMapper.Builder builder = new KNNVectorFieldMapper.Builder("test-field-name-1");

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

        builder.knnMethodContext.setValue(new KNNMethodContext(KNNEngine.DEFAULT, spaceType,
                new MethodComponentContext(METHOD_HNSW, ImmutableMap.of(METHOD_PARAMETER_M, m,
                        METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction))));

        builder.modelContext.setValue(new ModelContext("Random modelId", KNNEngine.DEFAULT, spaceType));

        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper instanceof KNNVectorFieldMapper.MethodFieldMapper);
        assertNotNull(knnVectorFieldMapper.knnMethod);
        assertNull(knnVectorFieldMapper.modelContext);
    }

    public void testBuilder_build_fromModel() {
        // Check that modelContext takes precedent over legacy
        KNNVectorFieldMapper.Builder builder = new KNNVectorFieldMapper.Builder("test-field-name-1");

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

        builder.modelContext.setValue(new ModelContext("Random modelId", KNNEngine.DEFAULT, spaceType));

        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper instanceof KNNVectorFieldMapper.ModelFieldMapper);
        assertNotNull(knnVectorFieldMapper.modelContext);
        assertNull(knnVectorFieldMapper.knnMethod);
    }

    public void testBuilder_build_fromLegacy() {
        // Check legacy is picked up if model context and method context are not set
        KNNVectorFieldMapper.Builder builder = new KNNVectorFieldMapper.Builder("test-field-name-1");

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

        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper instanceof KNNVectorFieldMapper.LegacyFieldMapper);

        assertNull(knnVectorFieldMapper.modelContext);
        assertNull(knnVectorFieldMapper.knnMethod);
    }

    public void testTypeParser_parse_fromKnnMethodContext() throws IOException {
        // Check that knnMethodContext is set
        String fieldName = "test-field-name";
        String indexName = "test-index-name";

        Settings settings = Settings.builder()
                .put(settings(CURRENT).build())
                .build();

        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser();

        int efConstruction = 321;
        int dimension = 133;
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder().startObject()
                .field("type", "knn_vector")
                .field("dimension", dimension)
                .startObject(KNN_METHOD)
                .field(NAME, METHOD_HNSW)
                .startObject(PARAMETERS)
                .field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction)
                .endObject()
                .endObject()
                .endObject();

        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(fieldName,
                xContentBuilderToMap(xContentBuilder), buildParserContext(indexName, settings));

        assertEquals(METHOD_HNSW, builder.knnMethodContext.get().getMethodComponent().getName());
        assertEquals(efConstruction, builder.knnMethodContext.get().getMethodComponent().getParameters()
                .get(METHOD_PARAMETER_EF_CONSTRUCTION));

        // Test invalid parameter
        XContentBuilder xContentBuilder2 = XContentFactory.jsonBuilder().startObject()
                .field("type", "knn_vector")
                .field("dimension", dimension)
                .startObject(KNN_METHOD)
                .field(NAME, METHOD_HNSW)
                .startObject(PARAMETERS)
                .field("invalid", "invalid")
                .endObject()
                .endObject()
                .endObject();

        expectThrows(ValidationException.class, () -> typeParser.parse(fieldName,
                xContentBuilderToMap(xContentBuilder2), buildParserContext(indexName, settings)));

        // Test invalid method
        XContentBuilder xContentBuilder3 = XContentFactory.jsonBuilder().startObject()
                .field("type", "knn_vector")
                .field("dimension", dimension)
                .startObject(KNN_METHOD)
                .field(NAME, "invalid")
                .endObject()
                .endObject();

        expectThrows(IllegalArgumentException.class, () -> typeParser.parse(fieldName,
                xContentBuilderToMap(xContentBuilder3), buildParserContext(indexName, settings)));

        // Test missing required parameter: dimension
        XContentBuilder xContentBuilder4 = XContentFactory.jsonBuilder().startObject()
                .field("type", "knn_vector").endObject();

        expectThrows(IllegalArgumentException.class, () -> typeParser.parse(fieldName,
                xContentBuilderToMap(xContentBuilder4), buildParserContext(indexName, settings)));

        // Check that this fails if model id is also set
        XContentBuilder xContentBuilder5 = XContentFactory.jsonBuilder().startObject()
                .field("type", "knn_vector")
                .field("dimension", dimension)
                .startObject(MODEL)
                .field(MODEL_ID, "test-id")
                .field(KNN_ENGINE, KNNEngine.DEFAULT.getName())
                .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.DEFAULT.getValue())
                .endObject()
                .startObject(KNN_METHOD)
                .field(NAME, METHOD_HNSW)
                .startObject(PARAMETERS)
                .field("invalid", "invalid")
                .endObject()
                .endObject()
                .endObject();

        expectThrows(IllegalArgumentException.class, () -> typeParser.parse(fieldName,
                xContentBuilderToMap(xContentBuilder5), buildParserContext(indexName, settings)));
    }

    public void testTypeParser_parse_fromModel() throws IOException {
        // Check that modelContext is set for the builder
        String fieldName = "test-field-name";
        String indexName = "test-index-name";

        Settings settings = Settings.builder()
                .put(settings(CURRENT).build())
                .build();

        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser();

        int dimension = 122;
        String modelId = "test-id";
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        SpaceType spaceType = SpaceType.L2;
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder().startObject()
                .field("type", "knn_vector")
                .field("dimension", dimension)
                .startObject(MODEL)
                .field(MODEL_ID, modelId)
                .field(KNN_ENGINE, knnEngine.getName())
                .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
                .endObject()
                .endObject();

        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(fieldName,
                xContentBuilderToMap(xContentBuilder), buildParserContext(indexName, settings));

        assertEquals(modelId, builder.modelContext.get().getModelId());
        assertEquals(spaceType, builder.modelContext.get().getSpaceType());
        assertEquals(knnEngine, builder.modelContext.get().getKNNEngine());
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

        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser();

        int dimension = 122;
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder().startObject()
                .field("type", "knn_vector")
                .field("dimension", dimension)
                .endObject();

        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(fieldName,
                xContentBuilderToMap(xContentBuilder), buildParserContext(indexName, settings));

        assertNull(builder.modelContext.get());
        assertNull(builder.knnMethodContext.get());
    }

    public void testKNNVectorFieldMapper_merge_fromKnnMethodContext() throws IOException {
        String fieldName = "test-field-name";
        String indexName = "test-index-name";

        Settings settings = Settings.builder()
                .put(settings(CURRENT).build())
                .build();

        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser();

        int dimension = 133;
        int efConstruction = 321;
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder().startObject()
                .field("type", "knn_vector")
                .field("dimension", dimension)
                .startObject(KNN_METHOD)
                .field(NAME, METHOD_HNSW)
                .startObject(PARAMETERS)
                .field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction)
                .endObject()
                .endObject()
                .endObject();

        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(fieldName,
                xContentBuilderToMap(xContentBuilder), buildParserContext(indexName, settings));

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
        xContentBuilder = XContentFactory.jsonBuilder().startObject()
                .field("type", "knn_vector")
                .field("dimension", dimension)
                .startObject(KNN_METHOD)
                .field(NAME, METHOD_HNSW)
                .endObject()
                .endObject();

        builder = (KNNVectorFieldMapper.Builder) typeParser.parse(fieldName, xContentBuilderToMap(xContentBuilder),
                buildParserContext(indexName, settings));
        KNNVectorFieldMapper knnVectorFieldMapper3 = builder.build(builderContext);
        expectThrows(IllegalArgumentException.class, () -> knnVectorFieldMapper1.merge(knnVectorFieldMapper3));
    }

    public void testKNNVectorFieldMapper_merge_fromModel() throws IOException {
        String fieldName = "test-field-name";
        String indexName = "test-index-name";

        Settings settings = Settings.builder()
                .put(settings(CURRENT).build())
                .build();

        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser();

        int dimension = 133;
        String modelId = "test-id";
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        SpaceType spaceType = SpaceType.DEFAULT;

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder().startObject()
                .field("type", "knn_vector")
                .field("dimension", dimension)
                .startObject(MODEL)
                .field(MODEL_ID, modelId)
                .field(KNN_ENGINE, knnEngine.getName())
                .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
                .endObject()
                .endObject();

        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(fieldName,
                xContentBuilderToMap(xContentBuilder), buildParserContext(indexName, settings));

        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper1 = builder.build(builderContext);


        // merge with itself - should be successful
        KNNVectorFieldMapper knnVectorFieldMapperMerge1 = (KNNVectorFieldMapper) knnVectorFieldMapper1.merge(knnVectorFieldMapper1);
        assertEquals(knnVectorFieldMapper1.modelContext, knnVectorFieldMapperMerge1.modelContext);

        // merge with another mapper of the same field with same context
        KNNVectorFieldMapper knnVectorFieldMapper2 = builder.build(builderContext);
        KNNVectorFieldMapper knnVectorFieldMapperMerge2 = (KNNVectorFieldMapper) knnVectorFieldMapper1.merge(knnVectorFieldMapper2);
        assertEquals(knnVectorFieldMapper1.modelContext, knnVectorFieldMapperMerge2.modelContext);

        // merge with another mapper of the same field with different context
        xContentBuilder = XContentFactory.jsonBuilder().startObject()
                .field("type", "knn_vector")
                .field("dimension", dimension)
                .startObject(KNN_METHOD)
                .field(NAME, METHOD_HNSW)
                .endObject()
                .endObject();

        builder = (KNNVectorFieldMapper.Builder) typeParser.parse(fieldName, xContentBuilderToMap(xContentBuilder),
                buildParserContext(indexName, settings));
        KNNVectorFieldMapper knnVectorFieldMapper3 = builder.build(builderContext);
        expectThrows(IllegalArgumentException.class, () -> knnVectorFieldMapper1.merge(knnVectorFieldMapper3));
    }

    public IndexMetadata buildIndexMetaData(String indexName, Settings settings) {
        return IndexMetadata.builder(indexName).settings(settings)
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
        IndexSettings indexSettings = new IndexSettings(buildIndexMetaData(indexName, settings), Settings.EMPTY,
                new IndexScopedSettings(Settings.EMPTY, new HashSet<>(IndexScopedSettings.BUILT_IN_INDEX_SETTINGS)));
        MapperService mapperService = mock(MapperService.class);
        when(mapperService.getIndexSettings()).thenReturn(indexSettings);

        // Setup blank
        return new Mapper.TypeParser.ParserContext(null, mapperService,
                type -> new KNNVectorFieldMapper.TypeParser(), CURRENT, null, null, null);

    }
}

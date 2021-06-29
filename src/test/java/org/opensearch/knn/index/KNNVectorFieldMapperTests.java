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

import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.Version.CURRENT;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class KNNVectorFieldMapperTests extends KNNTestCase {
    /**
     * Test that we can successfully create builder and get the relevant values. Note that parse needs to be called
     * in order to set the relevant parameters. Without calling parse, only the defaults will be set
     */
    public void testBuilder_build() {
        KNNVectorFieldMapper.Builder builder = new KNNVectorFieldMapper.Builder("test-field-name-1");

        // For default settings, everything in KNNVectorFieldMapper should be default after calling build
        Settings settings = Settings.builder()
                .put(settings(CURRENT).build())
                .build();
        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);

        assertNotNull(knnVectorFieldMapper);
        assertEquals(SpaceType.DEFAULT.getValue(), knnVectorFieldMapper.spaceType);
        assertEquals(KNNEngine.DEFAULT.getMethod(METHOD_HNSW).getMethodComponent().getParameters()
                .get(METHOD_PARAMETER_M).getDefaultValue().toString(), knnVectorFieldMapper.m);
        assertEquals(KNNEngine.DEFAULT.getMethod(METHOD_HNSW).getMethodComponent().getParameters()
                        .get(METHOD_PARAMETER_EF_CONSTRUCTION).getDefaultValue().toString(),
                knnVectorFieldMapper.efConstruction);


        // When passing spaceType, efConstruction and m settings, these should be set. This only applies to nmslib.
        // By default, the nmslib engine is used, so we do not have to configure it.
        String spaceType = SpaceType.COSINESIMIL.getValue();
        int m = 111;
        int efConstruction = 192;

        builder = new KNNVectorFieldMapper.Builder("test-field-name-2");

        settings = Settings.builder()
                .put(settings(CURRENT).build())
                .put(KNNSettings.KNN_SPACE_TYPE, spaceType)
                .put(KNNSettings.KNN_ALGO_PARAM_M, m)
                .put(KNNSettings.KNN_ALGO_PARAM_EF_CONSTRUCTION, efConstruction)
                .build();
        builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        knnVectorFieldMapper = builder.build(builderContext);

        assertEquals(spaceType, knnVectorFieldMapper.spaceType);
        assertEquals(Integer.toString(m), knnVectorFieldMapper.m);
        assertEquals(Integer.toString(efConstruction), knnVectorFieldMapper.efConstruction);

        // Test that mapping parameters get precedence over index settings
        //TODO: flip
        int m1 = 1000;
        int efConstruction1 = 12;
        SpaceType spaceType1 = SpaceType.L1;
        builder = new KNNVectorFieldMapper.Builder("test-field-name-3");
        builder.knnMethodContext.setValue(new KNNMethodContext(KNNEngine.NMSLIB, spaceType1,
                new MethodComponentContext(METHOD_HNSW,
                        ImmutableMap.of(METHOD_PARAMETER_M, m1, METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction1)
                )
        ));

        settings = Settings.builder()
                .put(settings(CURRENT).build())
                .put(KNNSettings.KNN_SPACE_TYPE, spaceType)
                .put(KNNSettings.KNN_ALGO_PARAM_M, m)
                .put(KNNSettings.KNN_ALGO_PARAM_EF_CONSTRUCTION, efConstruction)
                .build();
        builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        knnVectorFieldMapper = builder.build(builderContext);

        assertEquals(spaceType1.getValue(), knnVectorFieldMapper.spaceType);
        assertEquals(Integer.toString(m1), knnVectorFieldMapper.m);
        assertEquals(Integer.toString(efConstruction1), knnVectorFieldMapper.efConstruction);

        // When settings are empty, mapping parameters are used
        builder = new KNNVectorFieldMapper.Builder("test-field-name-4");
        builder.knnMethodContext.setValue(new KNNMethodContext(KNNEngine.NMSLIB, spaceType1,
                new MethodComponentContext(METHOD_HNSW,
                        ImmutableMap.of(METHOD_PARAMETER_M, m1, METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction1)
                )
        ));

        settings = Settings.builder()
                .put(settings(CURRENT).build())
                .build();
        builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        knnVectorFieldMapper = builder.build(builderContext);

        assertEquals(spaceType1.getValue(), knnVectorFieldMapper.spaceType);
        assertEquals(Integer.toString(m1), knnVectorFieldMapper.m);
        assertEquals(Integer.toString(efConstruction1), knnVectorFieldMapper.efConstruction);

        // Test builder for faiss
        builder = new KNNVectorFieldMapper.Builder("test-field-name-5");
        builder.knnMethodContext.setValue(new KNNMethodContext(KNNEngine.FAISS, spaceType1,
                new MethodComponentContext(METHOD_HNSW,
                        ImmutableMap.of(METHOD_PARAMETER_M, m1, METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction1)
                )
        ));

        settings = Settings.builder()
                .put(settings(CURRENT).build())
                .build();
        builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        knnVectorFieldMapper = builder.build(builderContext);

        assertEquals(builder.knnMethodContext.getValue(), knnVectorFieldMapper.knnMethod);
    }

    /**
     * Test that the builder correctly returns the parameters on call to getParameters
     */
    public void testBuilder_getParameters() {
        String fieldName = "test-field-name";
        KNNVectorFieldMapper.Builder builder = new KNNVectorFieldMapper.Builder(fieldName);
        assertEquals(5, builder.getParameters().size());
    }

    /**
     * Check that type parsing works for nmslib methods
     */
    public void testTypeParser_nmslib() throws IOException {
        String fieldName = "test-field-name";
        String indexName = "test-index-name";

        Settings settings = Settings.builder()
                .put(settings(CURRENT).build())
                .build();

        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser();

        int dimension = 133;
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder().startObject()
                .field("type", "knn_vector")
                .field("dimension", dimension)
                .endObject();

        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(fieldName,
                xContentBuilderToMap(xContentBuilder), buildParserContext(indexName, settings));

        assertEquals(dimension, builder.dimension.get().intValue());
        assertNull(builder.knnMethodContext.getValue()); // Defaults to null

        // Now, we need to test a custom parser
        int efConstruction = 321;
        xContentBuilder = XContentFactory.jsonBuilder().startObject()
                .field("type", "knn_vector")
                .field("dimension", dimension)
                .startObject(KNN_METHOD)
                .field(NAME, METHOD_HNSW)
                .startObject(PARAMETERS)
                .field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction)
                .endObject()
                .endObject()
                .endObject();

        builder = (KNNVectorFieldMapper.Builder) typeParser.parse(fieldName,
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
    }

    public void testMerge() throws IOException {
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

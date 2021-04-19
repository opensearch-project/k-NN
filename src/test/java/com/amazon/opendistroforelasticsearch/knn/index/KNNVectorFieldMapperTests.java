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

package com.amazon.opendistroforelasticsearch.knn.index;

import com.amazon.opendistroforelasticsearch.knn.KNNTestCase;
import com.amazon.opendistroforelasticsearch.knn.common.KNNConstants;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.bytes.BytesReference;
import org.opensearch.common.settings.IndexScopedSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.fielddata.IndexFieldData;
import org.opensearch.index.mapper.ContentPath;
import org.opensearch.index.mapper.Mapper;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.search.aggregations.support.CoreValuesSourceType;

import java.io.IOException;
import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static com.amazon.opendistroforelasticsearch.knn.index.KNNSettings.INDEX_KNN_SPACE_TYPE;
import static com.amazon.opendistroforelasticsearch.knn.index.KNNSettings.INDEX_KNN_ALGO_PARAM_M_SETTING;
import static com.amazon.opendistroforelasticsearch.knn.index.KNNSettings.INDEX_KNN_ALGO_PARAM_EF_CONSTRUCTION_SETTING;
import static org.opensearch.Version.CURRENT;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class KNNVectorFieldMapperTests extends KNNTestCase {

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

    public Map<String, Object> buildKnnNodeMap(int dimension) throws IOException {
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder().startObject()
                .field("type", "knn_vector")
                .field("dimension", dimension)
                .endObject();
        return XContentHelper.convertToMap(BytesReference.bytes(xContentBuilder), true,
                xContentBuilder.contentType()).v2();
    }

    public void testBuildKNNIndexSettings_normal() throws IOException {
        String indexName = "test-index";
        String fieldName = "test-field-name";
        int m = 73;
        int efConstruction = 47;
        int dimension = 100;

        Settings settings = Settings.builder()
                .put(settings(CURRENT).build())
                .put(KNNSettings.KNN_SPACE_TYPE, KNNConstants.COSINESIMIL)
                .put(KNNSettings.KNN_ALGO_PARAM_M, m)
                .put(KNNSettings.KNN_ALGO_PARAM_EF_CONSTRUCTION, efConstruction)
                .build();
        IndexMetadata indexMetadata = buildIndexMetaData(indexName, settings);

        Set<Setting<?>> settingSet = new HashSet<>(IndexScopedSettings.BUILT_IN_INDEX_SETTINGS);
        settingSet.add(INDEX_KNN_SPACE_TYPE);
        settingSet.add(INDEX_KNN_ALGO_PARAM_M_SETTING);
        settingSet.add(INDEX_KNN_ALGO_PARAM_EF_CONSTRUCTION_SETTING);

        IndexSettings indexSettings = new IndexSettings(indexMetadata, Settings.EMPTY,
                new IndexScopedSettings(Settings.EMPTY, settingSet));

        MapperService mapperService = mock(MapperService.class);
        when(mapperService.getIndexSettings()).thenReturn(indexSettings);

        Mapper.TypeParser.ParserContext context = new Mapper.TypeParser.ParserContext(null,
                mapperService, type -> new KNNVectorFieldMapper.TypeParser(), CURRENT, null,
                null, null);
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser();

        Map<String, Object> knnNodeMap = buildKnnNodeMap(dimension);
        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(fieldName, knnNodeMap,
                context);

        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);

        assertEquals(KNNConstants.COSINESIMIL, knnVectorFieldMapper.spaceType);
        assertEquals(String.valueOf(m), knnVectorFieldMapper.m);
        assertEquals(String.valueOf(efConstruction), knnVectorFieldMapper.efConstruction);
    }

    public void testBuildKNNIndexSettings_emptySettings() throws IOException {
        String indexName = "test-index";
        String fieldName = "test-field-name";
        int dimension = 100;

        Settings settings = Settings.builder()
                .put(settings(CURRENT).build())
                .build();
        IndexMetadata indexMetadata = buildIndexMetaData(indexName, settings);
        IndexSettings indexSettings = new IndexSettings(indexMetadata, Settings.EMPTY,
                new IndexScopedSettings(Settings.EMPTY, IndexScopedSettings.BUILT_IN_INDEX_SETTINGS));
        MapperService mapperService = mock(MapperService.class);
        when(mapperService.getIndexSettings()).thenReturn(indexSettings);

        Mapper.TypeParser.ParserContext context = new Mapper.TypeParser.ParserContext(null,
                mapperService, type -> new KNNVectorFieldMapper.TypeParser(), CURRENT, null,
                null, null);
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser();

        Map<String, Object> knnNodeMap = buildKnnNodeMap(dimension);
        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(fieldName, knnNodeMap,
                context);

        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);

        assertEquals(KNNSettings.INDEX_KNN_DEFAULT_SPACE_TYPE, knnVectorFieldMapper.spaceType);
        assertEquals(KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_M.toString(), knnVectorFieldMapper.m);
        assertEquals(KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION.toString(),
                knnVectorFieldMapper.efConstruction);
    }

    public void testVectorFieldMapperTypeFieldDataBuilder() {

        String mockIndexFieldName = "test-field-name";
        KNNVectorFieldMapper.KNNVectorFieldType vectorFieldType = new KNNVectorFieldMapper.KNNVectorFieldType(
                mockIndexFieldName, Collections.<String, String>emptyMap(), 10
        );
        IndexFieldData.Builder builder = vectorFieldType.fielddataBuilder(mockIndexFieldName, null);
        IndexFieldData<?> knnVectorIndexField = builder.build(null, null);
        assertNotNull(knnVectorIndexField);
        assertTrue(knnVectorIndexField instanceof KNNVectorIndexFieldData);
        assertEquals(mockIndexFieldName, knnVectorIndexField.getFieldName());
        assertEquals(CoreValuesSourceType.BYTES, knnVectorIndexField.getValuesSourceType());

    }
}

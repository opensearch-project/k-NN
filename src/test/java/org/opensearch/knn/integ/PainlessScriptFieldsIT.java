/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.apache.http.util.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.integ.PainlessScriptHelper.MappingProperty;
import org.opensearch.script.Script;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static org.opensearch.knn.integ.PainlessScriptHelper.createMapping;

// PainlesScriptScoreIT already tests every similarity methods with different field type. Hence,
// we don't have to recreate all tests for script_fields. From implementation point of view,
// it is clear if similarity method is supported by script_score, then same is applicable for script_fields
// provided script_fields context is supported. Hence, we test for one similarity method to verify that script_fields
// context is supported by this plugin.
public final class PainlessScriptFieldsIT extends KNNRestTestCase {

    private static final String NUMERIC_INDEX_FIELD_NAME = "price";

    private void buildTestIndex(final Map<String, Float[]> knnDocuments) throws Exception {
        List<MappingProperty> properties = buildMappingProperties();
        buildTestIndex(knnDocuments, properties);
    }

    private void buildTestIndex(final Map<String, Float[]> knnDocuments, final List<MappingProperty> properties) throws Exception {
        createKnnIndex(INDEX_NAME, createMapping(properties));
        for (Map.Entry<String, Float[]> data : knnDocuments.entrySet()) {
            addKnnDoc(INDEX_NAME, data.getKey(), FIELD_NAME, data.getValue());
        }
    }

    private Map<String, Float[]> getKnnVectorTestData() {
        Map<String, Float[]> data = new HashMap<>();
        data.put("1", new Float[] { 100.0f, 1.0f });
        data.put("2", new Float[] { 99.0f, 2.0f });
        data.put("3", new Float[] { 97.0f, 3.0f });
        data.put("4", new Float[] { 98.0f, 4.0f });
        return data;
    }

    private Map<String, Float[]> getCosineTestData() {
        Map<String, Float[]> data = new HashMap<>();
        data.put("0", new Float[] { 1.0f, -1.0f });
        data.put("2", new Float[] { 1.0f, 1.0f });
        data.put("1", new Float[] { 1.0f, 0.0f });
        return data;
    }

    /*
     The doc['field'] will throw an error if field is missing from the mappings.
     */
    private List<MappingProperty> buildMappingProperties() {
        List<MappingProperty> properties = new ArrayList<>();
        properties.add(MappingProperty.builder().name(FIELD_NAME).type(KNNVectorFieldMapper.CONTENT_TYPE).dimension("2").build());
        properties.add(MappingProperty.builder().name(NUMERIC_INDEX_FIELD_NAME).type("integer").build());
        return properties;
    }

    @SneakyThrows
    public void testCosineSimilarity_whenUsedInScriptFields_thenExecutesScript() {
        String source = String.format(Locale.ROOT, "1 + cosineSimilarity([2.0f, -2.0f], doc['%s'])", FIELD_NAME);
        String scriptFieldName = "similarity";
        Request request = buildPainlessScriptFieldsRequest(source, 3, getCosineTestData(), scriptFieldName);
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        List<KNNResult> results = parseSearchResponseScriptFields(EntityUtils.toString(response.getEntity()), scriptFieldName);
        assertEquals(3, results.size());

        String[] expectedDocIDs = { "0", "1", "2" };
        for (int i = 0; i < results.size(); i++) {
            assertEquals(expectedDocIDs[i], results.get(i).getDocId());
        }
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testGetValue_whenUsedInScriptFields_thenReturnsDocValues() {
        String source = String.format(Locale.ROOT, "doc['%s'].value[0]", FIELD_NAME);
        String scriptFieldName = "doc_value_field";
        Map<String, Float[]> testData = getKnnVectorTestData();
        Request request = buildPainlessScriptFieldsRequest(source, testData.size(), testData, scriptFieldName);

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        List<KNNResult> results = parseSearchResponseScriptFields(EntityUtils.toString(response.getEntity()), scriptFieldName);
        assertEquals(testData.size(), results.size());

        String[] expectedDocIDs = { "1", "2", "3", "4" };
        for (int i = 0; i < results.size(); i++) {
            assertEquals(expectedDocIDs[i], results.get(i).getDocId());
        }
        deleteKNNIndex(INDEX_NAME);
    }

    private Request buildPainlessScriptFieldsRequest(
        final String source,
        final int size,
        final Map<String, Float[]> documents,
        final String scriptFieldName
    ) throws Exception {
        buildTestIndex(documents);
        return constructScriptFieldsContextSearchRequest(
            INDEX_NAME,
            scriptFieldName,
            Collections.emptyMap(),
            Script.DEFAULT_SCRIPT_LANG,
            source,
            size,
            Collections.emptyMap()
        );
    }
}

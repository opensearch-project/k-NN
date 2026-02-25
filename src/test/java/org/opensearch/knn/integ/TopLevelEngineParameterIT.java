/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.common.Strings;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQueryBuilder;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.TOP_LEVEL_PARAMETER_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.NAME;

public class TopLevelEngineParameterIT extends KNNRestTestCase {
    private final static float[] TEST_VECTOR = new float[] { 1.0f, 2.0f };
    private final static int DIMENSION = 2;
    private final static int K = 1;
    private static final String INDEX_NAME = "top-level-engine-index";

    @SneakyThrows
    public void testBaseCase() {
        createTestIndexWithTopLevelEngineOnly();
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        deleteIndex(INDEX_NAME);

        createTestIndexWithMethodEngineOnly();
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        deleteIndex(INDEX_NAME);

        createTestIndexWithTopLevelEngineAndMethodEngine();
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        deleteIndex(INDEX_NAME);

        createTestIndexWithNoEngine();
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        deleteIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testDifferentEnginesDefined_ThenException() {
        Exception e = expectThrows(Exception.class, () -> createTestIndexWithDifferentEngines());
        assertTrue(e.getMessage(), e.getMessage().contains("Cannot specify conflicting engines: [faiss] and [lucene]"));
    }

    @SneakyThrows
    public void testEngineWithCompression() {
        // faiss, 2x compression -> valid
        createTestIndexWithCompression(KNNEngine.FAISS, "2x");
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        deleteIndex(INDEX_NAME);

        // faiss, 4x compression -> exception
        Exception e = expectThrows(Exception.class, () -> createTestIndexWithCompression(KNNEngine.FAISS, "4x"));
        assertTrue(e.getMessage(), e.getMessage().contains("Lucene is the only engine that supports 4x compression"));
        deleteIndex(INDEX_NAME);

        // lucene, 4x compression -> valid
        createTestIndexWithCompression(KNNEngine.LUCENE, "4x");
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        deleteIndex(INDEX_NAME);
    }

    private void createTestIndexWithTopLevelEngineOnly() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(TOP_LEVEL_PARAMETER_ENGINE, KNNEngine.LUCENE.getName())
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, mapping);

    }

    private void createTestIndexWithMethodEngineOnly() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, KNNEngine.LUCENE.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, mapping);
    }

    private void createTestIndexWithTopLevelEngineAndMethodEngine() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(TOP_LEVEL_PARAMETER_ENGINE, KNNEngine.LUCENE.getName())
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, KNNEngine.LUCENE.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, mapping);
    }

    private void createTestIndexWithNoEngine() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, mapping);
    }

    private void createTestIndexWithDifferentEngines() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(TOP_LEVEL_PARAMETER_ENGINE, KNNEngine.LUCENE.getName())
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, mapping);
    }

    private void createTestIndexWithCompression(KNNEngine engine, String compressionLevel) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(TOP_LEVEL_PARAMETER_ENGINE, engine.getName())
            .field("compression_level", compressionLevel)
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, mapping);
    }

    @Override
    public void validateKNNSearch(String testIndex, String testField, int dimension, int numDocs, int k) throws Exception {
        float[] queryVector = new float[dimension];
        Arrays.fill(queryVector, (float) numDocs);

        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder().k(k).fieldName(testField).vector(queryVector).build();
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject("query");
        knnQueryBuilder.doXContent(builder, ToXContent.EMPTY_PARAMS);
        builder.endObject().endObject();

        Response searchResponse = searchKNNIndex(testIndex, builder, k);
        assertOK(searchResponse);

        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), testField);

        assertEquals(k, results.size());
        for (int i = 0; i < k; i++) {
            assertEquals(numDocs - i - 1, Integer.parseInt(results.get(i).getDocId()));
        }
    }

    @Override
    protected void createKnnIndex(String index, String mapping) throws IOException {
        createIndex(index, getKNNDefaultIndexSettings(), null, null);
        putMappingRequest(index, mapping);
    }

    protected static void createIndex(String name, Settings settings, String mapping, String aliases) throws IOException {
        Request request = new Request("PUT", "/" + name);
        String entity = "{\"settings\": " + Strings.toString(MediaTypeRegistry.JSON, settings);
        if (mapping != null) {
            entity += ",\"mappings\" : {" + mapping + "}";
        }
        if (aliases != null) {
            entity += ",\"aliases\": {" + aliases + "}";
        }
        entity += "}";
        if (settings.getAsBoolean(IndexSettings.INDEX_SOFT_DELETES_SETTING.getKey(), true) == false) {
            expectSoftDeletesWarning(request, name);
        }
        request.setJsonEntity(entity);
        Response response = client().performRequest(request);
        assertOK(response);
    }
}

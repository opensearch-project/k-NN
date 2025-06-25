/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import com.google.common.collect.ImmutableList;
import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.Test;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.faiss.QFrameBitEncoder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE_KNN_VECTOR;

public class ADCIT extends KNNRestTestCase {

    private static final String TEST_FIELD_NAME = "test-field";

    private void makeOnlyQBitIndex(String indexName, String name, int dimension, int bits, boolean isUnderTest, SpaceType spaceType)
        throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(TEST_FIELD_NAME)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .startObject(KNN_METHOD)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .field(NAME, METHOD_HNSW)
            .startObject(PARAMETERS)
            .startObject("encoder")
            .field(NAME, "binary")
            .startObject("parameters")
            .field("bits", bits)
            .field(name, isUnderTest)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        createKnnIndex(indexName, builder.toString());
    }

    @Test
    public void testADCWithL2() {
        adcTestSpaceType(SpaceType.L2);
    }

    @Test
    public void testADCWithInnerProduct() {
        adcTestSpaceType(SpaceType.INNER_PRODUCT);
    }

    @Test
    public void testADCWithCosineSim() {
        adcTestSpaceType(SpaceType.COSINESIMIL);
    }

    @SneakyThrows
    private void adcTestSpaceType(SpaceType spaceType) {
        int dimension = 8;
        int bits = 1;
        int k = 10;

        // Generate 10 random vectors that we'll reuse
        List<Float[]> vectors = new ArrayList<>();
        Random random = new Random(42);
        for (int i = 0; i < 10; i++) {
            Float[] vector = new Float[dimension];
            for (int j = 0; j < dimension; j++) {
                vector[j] = random.nextFloat();
            }
            vectors.add(vector);
        }

        // Create control index (with ADC disabled)
        String controlIndexName = "adc-it-control-index-" + spaceType.toString().toLowerCase();
        makeOnlyQBitIndex(controlIndexName, QFrameBitEncoder.ENABLE_ADC_PARAM, dimension, bits, false, spaceType);

        // Index documents
        for (int i = 0; i < 10; i++) {
            addKnnDoc(controlIndexName, String.valueOf(i + 1), ImmutableList.of(TEST_FIELD_NAME), ImmutableList.of(vectors.get(i)));
        }
        forceMergeKnnIndex(controlIndexName);

        // Create test index (with ADC enabled)
        String testIndexName = "adc-it-test-index-" + spaceType.toString().toLowerCase();
        makeOnlyQBitIndex(testIndexName, QFrameBitEncoder.ENABLE_ADC_PARAM, dimension, bits, true, spaceType);

        // Index same vectors
        for (int i = 0; i < 10; i++) {
            addKnnDoc(testIndexName, String.valueOf(i + 1), ImmutableList.of(TEST_FIELD_NAME), ImmutableList.of(vectors.get(i)));
        }
        forceMergeKnnIndex(testIndexName);

        // Query builder for both control and test searches
        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("query")
                .startObject("knn")
                .startObject(TEST_FIELD_NAME)
                .array("vector", vectors.get(0))
                .field("k", k)
                .endObject()
                .endObject()
                .endObject()
                .endObject();

        // Search control index
        String controlResponse = EntityUtils.toString(searchKNNIndex(controlIndexName, queryBuilder, k).getEntity());
        List<Object> controlHits = parseSearchResponseHits(controlResponse);

        // Search test index
        String testResponse = EntityUtils.toString(searchKNNIndex(testIndexName, queryBuilder, k).getEntity());
        List<Object> testHits = parseSearchResponseHits(testResponse);

        assertEquals(10, controlHits.size());

        // Extract scores
        Double controlFirstHitScore = ((Double) (((Map<String, Object>) controlHits.get(0)).get("_score")));
        Double testFirstScore = ((Double) (((Map<String, Object>) testHits.get(0)).get("_score")));

        // For ADC test, scores should be different
        assertNotEquals(controlFirstHitScore, testFirstScore);

        deleteKNNIndex(controlIndexName);
        deleteKNNIndex(testIndexName);
    }

    @SneakyThrows
    private void adcFilterTestSpaceType(SpaceType spaceType) {
        int dimension = 8;
        int bits = 1;
        int k = 10;
        // Generate 10 random vectors that we'll reuse
        List<Float[]> vectors = new ArrayList<>();
        Random random = new Random(42);
        for (int i = 0; i < 10; i++) {
            Float[] vector = new Float[dimension];
            for (int j = 0; j < dimension; j++) {
                vector[j] = random.nextFloat();
            }
            vectors.add(vector);
        }

        // Create control index (without filter)
        String controlIndexName = "control-index" + spaceType.toString().toLowerCase();
        makeOnlyQBitIndex(controlIndexName, QFrameBitEncoder.ENABLE_ADC_PARAM, dimension, bits, true, spaceType);

        // Index documents
        for (int i = 0; i < 10; i++) {
            addKnnDoc(controlIndexName, String.valueOf(i + 1), ImmutableList.of(TEST_FIELD_NAME), ImmutableList.of(vectors.get(i)));
        }
        forceMergeKnnIndex(controlIndexName);

        // Search without filter
        XContentBuilder controlQueryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(TEST_FIELD_NAME)
            .array("vector", vectors.get(0))
            .field("k", k)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String controlResponse = EntityUtils.toString(searchKNNIndex(controlIndexName, controlQueryBuilder, k).getEntity());
        List<Object> controlHits = parseSearchResponseHits(controlResponse);
        List<Double> controlScores = controlHits.stream()
            .map(hit -> (Double) ((Map<String, Object>) hit).get("_score"))
            .collect(Collectors.toList());

        // Create test index (with filter)
        String testIndexName = "test-index" + spaceType.toString().toLowerCase();
        makeOnlyQBitIndex(testIndexName, QFrameBitEncoder.ENABLE_ADC_PARAM, dimension, bits, true, spaceType);

        // Index same vectors
        for (int i = 0; i < 10; i++) {
            addKnnDoc(testIndexName, String.valueOf(i + 1), ImmutableList.of(TEST_FIELD_NAME), ImmutableList.of(vectors.get(i)));
        }
        forceMergeKnnIndex(testIndexName);

        // Search with match_all filter
        XContentBuilder testQueryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(TEST_FIELD_NAME)
            .array("vector", vectors.get(0))
            .field("k", k)
            .startObject("filter")
            .startObject("match_all")
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String testResponse = EntityUtils.toString(searchKNNIndex(testIndexName, testQueryBuilder, k).getEntity());
        List<Object> testHits = parseSearchResponseHits(testResponse);
        List<Double> testScores = testHits.stream()
            .map(hit -> (Double) ((Map<String, Object>) hit).get("_score"))
            .collect(Collectors.toList());

        // Assert that hits are the same
        assertEquals("Number of hits should be equal", controlScores.size(), testScores.size());

        for (int i = 0; i < controlScores.size(); i++) {
            assertEquals("Scores should be equal at position " + i, controlScores.get(i), testScores.get(i), 0.0001);
        }

        // Verify same document IDs and order
        List<String> controlIds = controlHits.stream()
            .map(hit -> (String) ((Map<String, Object>) hit).get("_id"))
            .collect(Collectors.toList());
        List<String> testIds = testHits.stream().map(hit -> (String) ((Map<String, Object>) hit).get("_id")).collect(Collectors.toList());

        assertEquals("Document IDs should be in the same order", controlIds, testIds);
        deleteKNNIndex(controlIndexName);
        deleteKNNIndex(testIndexName);
    }

    @SneakyThrows
    public void testFilterADC() {
        /*
        0. for each of control, test:
        1. create index. ingest 10 documents. force merge index.
        2. run with match all filter query and k = 10
        3. Create (adc) index. ingest the same 10 vectors, but with different document ids (11 to 20).
        4. assert that the scores of the results are the same in both searches.src/test/java/org/opensearch/knn/index/ADCIT.java
         */
        for (SpaceType spaceType : new SpaceType[] { SpaceType.L2, SpaceType.INNER_PRODUCT, SpaceType.COSINESIMIL }) {
            adcFilterTestSpaceType(spaceType);
        }
    }

    protected List<Object> parseSearchResponseHits(String responseBody) throws IOException {
        return (List<Object>) ((Map<String, Object>) createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map()
            .get("hits")).get("hits");
    }
}

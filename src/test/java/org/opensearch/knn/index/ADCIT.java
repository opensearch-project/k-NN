/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import java.util.Locale;

import com.google.common.collect.ImmutableList;
import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.Test;
import org.opensearch.common.settings.Settings;
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
import static org.opensearch.knn.index.KNNSettings.MEMORY_OPTIMIZED_KNN_SEARCH_MODE;

public class ADCIT extends KNNRestTestCase {

    private static final String TEST_FIELD_NAME = "test-field";

    private XContentBuilder qBitMapping(String name, int dimension, int bits, boolean isUnderTest, SpaceType spaceType) throws IOException {
        return XContentFactory.jsonBuilder()
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
    }

    private void makeOnlyQBitIndex(String indexName, String name, int dimension, int bits, boolean isUnderTest, SpaceType spaceType)
        throws IOException {
        createKnnIndex(indexName, qBitMapping(name, dimension, bits, isUnderTest, spaceType).toString());
    }

    private void makeOnlyQBitIndex(
        String indexName,
        String name,
        int dimension,
        int bits,
        boolean isUnderTest,
        SpaceType spaceType,
        boolean memoryOptimized
    ) throws IOException {
        final String mapping = qBitMapping(name, dimension, bits, isUnderTest, spaceType).toString();
        if (memoryOptimized) {
            final Settings settings = Settings.builder().put("index.knn", true).put(MEMORY_OPTIMIZED_KNN_SEARCH_MODE, true).build();
            createKnnIndex(indexName, settings, mapping);
        } else {
            createKnnIndex(indexName, mapping);
        }
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

    /**
     * ADC cosine scoring with memory-optimized search (MOS) enabled must agree with the standard
     * (MOS-disabled, native Faiss) path.
     *
     * <p>Regression guard for the ADC cosine fix: the memory-optimized ADC path emits MaxIP-format
     * scores and relies on a MaxIP -> cosine post-conversion in MemoryOptimizedKNNWeight. When that
     * conversion was inadvertently removed, MOS returned roughly 2x the standard value. Neither ADCIT
     * nor the MOS suite exercised ADC + cosine with MOS enabled, so no integration test covered this
     * branch.</p>
     */
    @Test
    @SneakyThrows
    public void testADCWithCosineSim_whenMemoryOptimized_thenMatchesStandardScores() {
        final int dimension = 8;
        final int bits = 1;
        final int k = 10;
        final SpaceType spaceType = SpaceType.COSINESIMIL;

        // Generate 10 random vectors that we'll reuse across both indices.
        List<Float[]> vectors = new ArrayList<>();
        Random random = new Random(42);
        for (int i = 0; i < 10; i++) {
            Float[] vector = new Float[dimension];
            for (int j = 0; j < dimension; j++) {
                vector[j] = random.nextFloat();
            }
            vectors.add(vector);
        }

        // Standard index: ADC enabled, MOS disabled -> native Faiss search path (DefaultKNNWeight).
        String standardIndexName = "adc-cosine-standard";
        makeOnlyQBitIndex(standardIndexName, QFrameBitEncoder.ENABLE_ADC_PARAM, dimension, bits, true, spaceType, false);

        // Memory-optimized index: ADC enabled, MOS enabled -> MemoryOptimizedKNNWeight ADC branch.
        String memoryOptimizedIndexName = "adc-cosine-memory-optimized";
        makeOnlyQBitIndex(memoryOptimizedIndexName, QFrameBitEncoder.ENABLE_ADC_PARAM, dimension, bits, true, spaceType, true);

        for (int i = 0; i < 10; i++) {
            addKnnDoc(standardIndexName, String.valueOf(i + 1), ImmutableList.of(TEST_FIELD_NAME), ImmutableList.of(vectors.get(i)));
            addKnnDoc(memoryOptimizedIndexName, String.valueOf(i + 1), ImmutableList.of(TEST_FIELD_NAME), ImmutableList.of(vectors.get(i)));
        }
        forceMergeKnnIndex(standardIndexName);
        forceMergeKnnIndex(memoryOptimizedIndexName);

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

        String standardResponse = EntityUtils.toString(searchKNNIndex(standardIndexName, queryBuilder, k).getEntity());
        List<Object> standardHits = parseSearchResponseHits(standardResponse);

        String memoryOptimizedResponse = EntityUtils.toString(searchKNNIndex(memoryOptimizedIndexName, queryBuilder, k).getEntity());
        List<Object> memoryOptimizedHits = parseSearchResponseHits(memoryOptimizedResponse);

        assertEquals(10, standardHits.size());
        assertEquals("MOS should return the same number of hits as standard search", standardHits.size(), memoryOptimizedHits.size());

        for (int i = 0; i < standardHits.size(); i++) {
            Map<String, Object> standardHit = (Map<String, Object>) standardHits.get(i);
            Map<String, Object> memoryOptimizedHit = (Map<String, Object>) memoryOptimizedHits.get(i);

            assertEquals("Doc order should match at position " + i, standardHit.get("_id"), memoryOptimizedHit.get("_id"));

            double standardScore = (Double) standardHit.get("_score");
            double memoryOptimizedScore = (Double) memoryOptimizedHit.get("_score");

            // MOS must produce the same scores as the standard (native Faiss) path (documented invariant).
            // This is the regression guard for the ADC cosine fix: without the MaxIP -> cosine post-conversion
            // in MemoryOptimizedKNNWeight, MOS returned the raw MaxIP score (~2x the standard value).
            //
            // Note: 1-bit ADC reconstructs data vectors that are not unit-norm, so the asymmetric cosine
            // score can legitimately exceed 1.0 on both the standard and MOS paths -- we therefore assert
            // parity with the standard path rather than an absolute [0, 1] bound. The epsilon comfortably
            // tolerates minor native-vs-Java float differences while still catching the 2x regression.
            assertEquals("Scores should match at position " + i, standardScore, memoryOptimizedScore, 0.05);
        }

        deleteKNNIndex(standardIndexName);
        deleteKNNIndex(memoryOptimizedIndexName);
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
        String controlIndexName = "adc-it-control-index-" + spaceType.toString().toLowerCase(Locale.ROOT);
        makeOnlyQBitIndex(controlIndexName, QFrameBitEncoder.ENABLE_ADC_PARAM, dimension, bits, false, spaceType);

        // Index documents
        for (int i = 0; i < 10; i++) {
            addKnnDoc(controlIndexName, String.valueOf(i + 1), ImmutableList.of(TEST_FIELD_NAME), ImmutableList.of(vectors.get(i)));
        }
        forceMergeKnnIndex(controlIndexName);

        // Create test index (with ADC enabled)
        String testIndexName = "adc-it-test-index-" + spaceType.toString().toLowerCase(Locale.ROOT);
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
        String controlIndexName = "control-index" + spaceType.toString().toLowerCase(Locale.ROOT);
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
        String testIndexName = "test-index" + spaceType.toString().toLowerCase(Locale.ROOT);
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
        assertEquals("[" + spaceType + "] Number of hits should be equal", controlScores.size(), testScores.size());

        for (int i = 0; i < controlScores.size(); i++) {
            assertEquals("[" + spaceType + "] Scores should be equal at position " + i, controlScores.get(i), testScores.get(i), 0.0001);
        }

        // Verify same document IDs and order
        List<String> controlIds = controlHits.stream()
            .map(hit -> (String) ((Map<String, Object>) hit).get("_id"))
            .collect(Collectors.toList());
        List<String> testIds = testHits.stream().map(hit -> (String) ((Map<String, Object>) hit).get("_id")).collect(Collectors.toList());

        assertEquals("[" + spaceType + "] Document IDs should be in the same order", controlIds, testIds);
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

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.KNNCommonSettingsBuilder;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQueryBuilder;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;

@Log4j2
public class AASimilarityMultiShardTestIT extends KNNSingleNodeTestCase {

    private final String spaceType;
    private final String engine;
    private final String groundTruthFile;
    private final Settings settings;

    public AASimilarityMultiShardTestIT(String spaceType, String engine, String groundTruthFile, Settings settings) {
        this.spaceType = spaceType;
        this.engine = engine;
        this.groundTruthFile = groundTruthFile;
        this.settings = settings;
    }

    @ParametersFactory
    public static Collection<Object[]> parameters() {
        return Arrays.asList(
            new Object[][] {
                {
                    SpaceType.COSINESIMIL.getValue(),
                    KNNEngine.FAISS.getName(),
                    "data/test_sanity_ground_truth_cosine_1000.json",
                    KNNCommonSettingsBuilder.defaultSettings().multiShard().build() },
                {
                    SpaceType.COSINESIMIL.getValue(),
                    KNNEngine.FAISS.getName(),
                    "data/test_sanity_ground_truth_cosine_1000.json",
                    KNNCommonSettingsBuilder.defaultSettings().memOptSearch().multiShard().build() } }
        );
    }

    public void testSimilaritySearchMultiShard() throws IOException, InterruptedException, ExecutionException {
        String indexName = "similarity-test-index";
        String fieldName = "vector";
        int dimensions = 16;

        log.info("111111111111111111111111111");
        createIndex(indexName, this.settings);
        log.info("222222222222222222222222222");
        createMapping(indexName, fieldName, dimensions);

        log.info("333333333333333333333333333");
        List<Object> groundTruth = loadGroundTruth(groundTruthFile);
        // TMP
        groundTruth = groundTruth.subList(0, 100);
        // TMP
        log.info("444444444444444444444444444");
        float[] queryVector = setupTestFromGroundTruth(indexName, fieldName, dimensions, groundTruth);
        log.info("555555555555555555555555555");

        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder().fieldName(fieldName).vector(queryVector).k(10).build();

        SearchResponse response = client().prepareSearch(indexName).setQuery(knnQueryBuilder).get();
        log.info("666666666666666666666666666");

        validateResults(response, groundTruth);
    }

    /**
     * Loads a ground truth file into a JSON list of objects.
     *
     * The ground truth file should be of the format:
     * [
     *   {
     *     "id": "338",
     *     "vector": [0.0, 1.0]
     *     "score": 1.0
     *   },
     *   {
     *     "id": "61",
     *     "vector": [1.0, 2.0]
     *     "score": 0.9637892
     *   },
     * ]
     * The first vector will be the query vector.
     * @param groundTruthFile
     * @return List of json vectors
     * @throws IOException
     */
    @SneakyThrows
    private List<Object> loadGroundTruth(String groundTruthFile) {
        final URL hnswWithOneVector = AASimilarityMultiShardTestIT.class.getClassLoader().getResource(groundTruthFile);
        try (InputStream inputStream = Files.newInputStream(Path.of(hnswWithOneVector.toURI()))) {
            if (inputStream == null) {
                throw new IOException("Resource not found: " + groundTruthFile);
            }
            XContentParser parser = JsonXContent.jsonXContent.createParser(null, null, inputStream);
            List<Object> groundTruth = parser.list();
            parser.close();
            return groundTruth;
        }
    }

    private void createMapping(String indexName, String fieldName, int dimensions) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimensions)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType)
            .field(KNN_ENGINE, engine)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        client().admin().indices().preparePutMapping(indexName).setSource(builder).get();
    }

    private void validateResults(SearchResponse response, List<Object> groundTruth) {
        assertEquals(10, response.getHits().getHits().length);
        for (int i = 0; i < 10; i++) {
            Map<String, Object> expectedDoc = (Map<String, Object>) groundTruth.get(i);
            String expectedId = (String) expectedDoc.get("id");
            float expectedScore = ((Number) expectedDoc.get("score")).floatValue();
            assertEquals(expectedId, response.getHits().getAt(i).getId());
            assertEquals(expectedScore, response.getHits().getAt(i).getScore(), 0.001f);
        }
    }
}

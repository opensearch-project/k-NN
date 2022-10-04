/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.opensearch.knn.TestUtils;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.index.query.TermQueryBuilder;

import org.opensearch.client.Request;
import org.opensearch.client.ResponseException;
import org.opensearch.common.Strings;
import org.opensearch.common.xcontent.ToXContent;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;

import java.io.IOException;

import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;

/**
 * Tests scenarios specific to filtering functionality in k-NN in case Lucene is set as an engine
 */
public class LuceneFilteringIT extends AbstractRollingUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 50;
    private static final int K = 10;
    private static final int NUM_DOCS = 100;
    private static final TermQueryBuilder TERM_QUERY = QueryBuilders.termQuery("_id", "100");

    public void testLuceneFiltering() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        float[] queryVector = TestUtils.getQueryVectors(1, DIMENSIONS, NUM_DOCS, true)[0];
        switch (getClusterType()) {
            case OLD:
                createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMappingWithLuceneField(TEST_FIELD, DIMENSIONS));
                bulkAddKnnDocs(testIndex, TEST_FIELD, TestUtils.getIndexVectors(NUM_DOCS, DIMENSIONS, true), NUM_DOCS);
                validateSearchKNNIndexFailed(testIndex, new KNNQueryBuilder(TEST_FIELD, queryVector, K, TERM_QUERY), K);
                break;
            case MIXED:
                validateSearchKNNIndexFailed(testIndex, new KNNQueryBuilder(TEST_FIELD, queryVector, K, TERM_QUERY), K);
                break;
            case UPGRADED:
                searchKNNIndex(testIndex, new KNNQueryBuilder(TEST_FIELD, queryVector, K, TERM_QUERY), K);
                deleteKNNIndex(testIndex);
                break;
        }
    }

    protected String createKnnIndexMappingWithLuceneField(final String fieldName, int dimension) throws IOException {
        return Strings.toString(
            XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(fieldName)
                .field("type", "knn_vector")
                .field("dimension", Integer.toString(dimension))
                .startObject("method")
                .field("name", "hnsw")
                .field("engine", "lucene")
                .field("space_type", "l2")
                .endObject()
                .endObject()
                .endObject()
                .endObject()
        );
    }

    private void validateSearchKNNIndexFailed(String index, KNNQueryBuilder knnQueryBuilder, int resultSize) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject("query");
        knnQueryBuilder.doXContent(builder, ToXContent.EMPTY_PARAMS);
        builder.endObject().endObject();

        Request request = new Request("POST", "/" + index + "/_search");

        request.addParameter("size", Integer.toString(resultSize));
        request.addParameter("explain", Boolean.toString(true));
        request.addParameter("search_type", "query_then_fetch");
        request.setJsonEntity(Strings.toString(builder));

        expectThrows(ResponseException.class, () -> client().performRequest(request));
    }
}

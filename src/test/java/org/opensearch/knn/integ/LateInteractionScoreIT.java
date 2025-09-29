/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;

import java.util.List;
import java.util.ArrayList;

public class LateInteractionScoreIT extends KNNRestTestCase {

    public void testLateInteractionScore() throws Exception {
        createIndexWithMapping();
        indexDocuments();

        String script = "lateInteractionScore(params.query_vector, 'my_vector', params._source)";

        List<List<Double>> queryVectors = new ArrayList<>();
        List<Double> qv1 = new ArrayList<>();
        qv1.add(0.1);
        qv1.add(0.2);
        queryVectors.add(qv1);

        Request request = new Request("POST", "/" + INDEX_NAME + "/_search");
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject("query");
        builder.startObject("script_score");
        builder.startObject("query");
        builder.startObject("match_all").endObject();
        builder.endObject();
        builder.startObject("script");
        builder.field("source", script);
        builder.startObject("params");
        builder.field("query_vector", queryVectors);
        builder.endObject();
        builder.endObject();
        builder.endObject();
        builder.endObject();
        builder.endObject();
        request.setJsonEntity(builder.toString());

        Response response = client().performRequest(request);
        assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        String responseBody = EntityUtils.toString(response.getEntity());
        // TODO: Add assertions to check the scores and order of documents
    }

    private void createIndexWithMapping() throws Exception {
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject("my_vector")
            .field("type", "object")
            .field("enabled", "false")
            .endObject()
            .endObject()
            .endObject()
            .toString();
        createKnnIndex(INDEX_NAME, mapping);
    }

    private void indexDocuments() throws Exception {
        List<List<Double>> docVectors1 = new ArrayList<>();
        List<Double> dv1 = new ArrayList<>();
        dv1.add(0.3);
        dv1.add(0.4);
        docVectors1.add(dv1);
        addKnnDoc(INDEX_NAME, "1", "my_vector", docVectors1);

        List<List<Double>> docVectors2 = new ArrayList<>();
        List<Double> dv2 = new ArrayList<>();
        dv2.add(0.1);
        dv2.add(0.2);
        docVectors2.add(dv2);
        addKnnDoc(INDEX_NAME, "2", "my_vector", docVectors2);

        flushIndex(INDEX_NAME);
    }
}

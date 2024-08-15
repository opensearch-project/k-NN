/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.opensearch.client.Request;
import org.opensearch.client.ResponseException;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;

import java.io.IOException;
import java.util.Locale;

import static org.opensearch.knn.index.query.parser.RescoreParser.RESCORE_OVERSAMPLE_PARAMETER;
import static org.opensearch.knn.index.query.parser.RescoreParser.RESCORE_PARAMETER;

public class QueryParseIT extends KNNRestTestCase {

    private final static float[] TEST_VECTOR = new float[] { 1.0f, 2.0f };
    private final static int DIMENSION = 2;
    private final static int K = 1;

    @SneakyThrows
    public void testRescore() {
        createTestIndex();
        assertValid(
            buildRequest(
                closeQueryXContentBuilder(
                    setupQueryXContentBuilder().field("vector", TEST_VECTOR).field("k", K).startObject("rescore").endObject()
                )
            )
        );

        assertValid(
            buildRequest(
                closeQueryXContentBuilder(
                    setupQueryXContentBuilder().field("vector", TEST_VECTOR)
                        .field("k", K)
                        .startObject(RESCORE_PARAMETER)
                        .field(RESCORE_OVERSAMPLE_PARAMETER, 2)
                        .endObject()
                )
            )
        );

        assertValid(
            buildRequest(
                closeQueryXContentBuilder(
                    setupQueryXContentBuilder().field("vector", TEST_VECTOR).field("k", K).startObject(RESCORE_PARAMETER).endObject()
                )
            )
        );

        assertValid(
            buildRequest(
                closeQueryXContentBuilder(
                    setupQueryXContentBuilder().field("vector", TEST_VECTOR).field("k", K).field(RESCORE_PARAMETER, true)
                )
            )
        );

        assertValid(
            buildRequest(
                closeQueryXContentBuilder(
                    setupQueryXContentBuilder().field("vector", TEST_VECTOR).field("k", K).field(RESCORE_PARAMETER, false)
                )
            )
        );

        // Invalid value for rescore
        assertInvalid(
            buildRequest(
                closeQueryXContentBuilder(
                    setupQueryXContentBuilder().field("vector", TEST_VECTOR).field("k", K).field(RESCORE_PARAMETER, "invalid")
                )
            )
        );

        // Invalid rescore param
        assertInvalid(
            buildRequest(
                closeQueryXContentBuilder(
                    setupQueryXContentBuilder().field("vector", TEST_VECTOR)
                        .field("k", K)
                        .startObject(RESCORE_OVERSAMPLE_PARAMETER)
                        .field("invalid_param", "invalid")
                        .endObject()
                )
            )
        );

        // Invalid rescore param value
        assertInvalid(
            buildRequest(
                closeQueryXContentBuilder(
                    setupQueryXContentBuilder().field("vector", TEST_VECTOR)
                        .field("k", K)
                        .startObject(RESCORE_PARAMETER)
                        .field(RESCORE_OVERSAMPLE_PARAMETER, "invalid")
                        .endObject()
                )
            )
        );
    }

    private XContentBuilder setupQueryXContentBuilder() throws IOException {
        return XContentFactory.jsonBuilder().startObject().startObject("query").startObject("knn").startObject(FIELD_NAME);
    }

    private XContentBuilder closeQueryXContentBuilder(XContentBuilder xContentBuilder) throws IOException {
        return xContentBuilder.endObject().endObject().endObject().endObject();
    }

    private void assertValid(Request request) throws IOException {
        assertOK(client().performRequest(request));
    }

    private void assertInvalid(Request request) {
        expectThrows(ResponseException.class, () -> client().performRequest(request));
    }

    private Request buildRequest(XContentBuilder xContentBuilder) {
        Request request = new Request("POST", String.format(Locale.ROOT, "/%s/_search", INDEX_NAME));
        request.addParameter("size", Integer.toString(10));
        request.addParameter("explain", Boolean.toString(true));
        request.setJsonEntity(xContentBuilder.toString());
        return request;
    }

    private void createTestIndex() throws IOException {
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
}

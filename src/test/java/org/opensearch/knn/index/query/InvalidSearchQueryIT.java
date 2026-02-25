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

package org.opensearch.knn.index.query;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import lombok.AllArgsConstructor;
import lombok.SneakyThrows;
import org.opensearch.client.Request;
import org.opensearch.client.ResponseException;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;

import static com.carrotsearch.randomizedtesting.RandomizedTest.$;
import static com.carrotsearch.randomizedtesting.RandomizedTest.$$;

@AllArgsConstructor
public class InvalidSearchQueryIT extends KNNRestTestCase {

    private String description;
    private XContentBuilder xContentBuilder;

    @ParametersFactory(argumentFormatting = "description:%1$s; request:%2$s, expectedexception:%3$s")
    public static Collection<Object[]> parameters() throws IOException {
        /**
         * Valid query:
         * {
         *    query: {
         *      knn: {
         *         test_field: {
         *             vector: [1.0, 2.0],
         *             k: 1,
         *             method_parameter: {
         *                ef_search: 10
         *             }
         *         }
         *      }
         *    }
         * }
         */

        return Arrays.asList(
            $$(
                $(
                    "Empty method_parameter",
                    XContentFactory.jsonBuilder()
                        .startObject()
                        .startObject("query")
                        .startObject("knn")
                        .startObject(FIELD_NAME)
                        .field("vector", new float[] { 1.0f, 2.0f })
                        .field("k", 1)
                        .startObject("method_parameter")
                        .endObject()
                        .endObject()
                        .endObject()
                        .endObject()
                        .endObject()
                ),
                $(
                    "ef_search string",
                    XContentFactory.jsonBuilder()
                        .startObject()
                        .startObject("query")
                        .startObject("knn")
                        .startObject(FIELD_NAME)
                        .field("vector", new float[] { 1.0f, 2.0f })
                        .field("k", 1)
                        .startObject("method_parameter")
                        .field("ef_search", "string value")
                        .endObject()
                        .endObject()
                        .endObject()
                        .endObject()
                        .endObject()
                ),
                $(
                    "ef_search less than 0",
                    XContentFactory.jsonBuilder()
                        .startObject()
                        .startObject("query")
                        .startObject("knn")
                        .startObject(FIELD_NAME)
                        .field("vector", new float[] { 1.0f, 2.0f })
                        .field("k", 1)
                        .startObject("method_parameter")
                        .field("ef_search", -1)
                        .endObject()
                        .endObject()
                        .endObject()
                        .endObject()
                        .endObject()
                ),
                $(
                    "nprobes string",
                    XContentFactory.jsonBuilder()
                        .startObject()
                        .startObject("query")
                        .startObject("knn")
                        .startObject(FIELD_NAME)
                        .field("vector", new float[] { 1.0f, 2.0f })
                        .field("k", 1)
                        .startObject("method_parameter")
                        .field("nprobes", "string value")
                        .endObject()
                        .endObject()
                        .endObject()
                        .endObject()
                        .endObject()
                ),
                $(
                    "nprobes less than 0",
                    XContentFactory.jsonBuilder()
                        .startObject()
                        .startObject("query")
                        .startObject("knn")
                        .startObject(FIELD_NAME)
                        .field("vector", new float[] { 1.0f, 2.0f })
                        .field("k", 1)
                        .startObject("method_parameter")
                        .field("nprobes", -10)
                        .endObject()
                        .endObject()
                        .endObject()
                        .endObject()
                        .endObject()
                )
            )
        );
    }

    @SneakyThrows
    public void testEndToEnd_whenMethodIsHNSWFlat_thenSucceed() {
        Request request = new Request("POST", "/dummy_index/_search");
        request.setJsonEntity(xContentBuilder.toString());

        request.addParameter("size", Integer.toString(10));
        request.addParameter("explain", Boolean.toString(true));
        request.addParameter("search_type", "query_then_fetch");

        expectThrows(ResponseException.class, () -> client().performRequest(request));
    }
}

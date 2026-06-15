/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.integ.PainlessScriptHelper.MappingProperty;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.opensearch.knn.integ.PainlessScriptHelper.createMapping;

/**
 * Verifies that Lucene Expression scripting is not supported for the knn_vector
 * field type. A script_score query using lang: "expression" against a knn_vector
 * field is rejected by the server with an error indicating the field must be
 * numeric, date, or geopoint, since knn_vector does not expose numeric doc values.
 * For comparison, the Painless equivalent (doc['my_vector'].value) is supported and
 * exercised in PainlessScriptScoreIT. This test guards that unsupported behavior so
 * the rejection stays explicit if the scripting paths change.
 */
public final class ExpressionScriptScoreIT extends KNNRestTestCase {

    private static final String EXPRESSION_LANG = "expression";

    private List<MappingProperty> buildKnnVectorMapping() {
        List<MappingProperty> properties = new ArrayList<>();
        properties.add(MappingProperty.builder().name(FIELD_NAME).type(KNNVectorFieldMapper.CONTENT_TYPE).dimension("2").build());
        return properties;
    }

    private void buildSingleDocIndex() throws Exception {
        createKnnIndex(INDEX_NAME, createMapping(buildKnnVectorMapping()));
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 1.0f });
    }

    /**
     * doc['my_vector'].value uses standard Expression field-access syntax. The
     * request is rejected with HTTP 400 and an error indicating the field must be
     * numeric, date, or geopoint, confirming Expression cannot operate on a
     * knn_vector field.
     */
    public void testExpressionScriptingUnsupportedOnKnnVector() throws Exception {
        buildSingleDocIndex();
        try {
            String source = String.format("doc['%s'].value", FIELD_NAME);
            QueryBuilder qb = new MatchAllQueryBuilder();
            Request request = constructScriptScoreContextSearchRequest(
                INDEX_NAME,
                qb,
                Collections.emptyMap(),
                EXPRESSION_LANG,
                source,
                1,
                Collections.emptyMap()
            );

            ResponseException e = expectThrows(ResponseException.class, () -> client().performRequest(request));
            Response response = e.getResponse();
            int statusCode = response.getStatusLine().getStatusCode();
            String body = EntityUtils.toString(response.getEntity());
            logger.info("Expression script_score on knn_vector rejected as expected: {}", response.getStatusLine());
            logger.info("Response body: {}", body);

            assertEquals("Expected HTTP 400 for lang=expression on knn_vector", 400, statusCode);
            assertTrue(
                "Expected error indicating the field must be numeric, date, or geopoint, but was: " + body,
                body.contains("must be numeric, date, or geopoint")
            );
        } finally {
            deleteKNNIndex(INDEX_NAME);
        }
    }
}

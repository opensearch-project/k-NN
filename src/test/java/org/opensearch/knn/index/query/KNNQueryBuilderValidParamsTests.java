/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import lombok.AllArgsConstructor;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.util.Arrays;
import java.util.Collection;
import java.util.Map;

import static com.carrotsearch.randomizedtesting.RandomizedTest.$;
import static com.carrotsearch.randomizedtesting.RandomizedTest.$$;

@AllArgsConstructor
public class KNNQueryBuilderValidParamsTests extends KNNTestCase {

    private static final float[] QUERY_VECTOR = new float[] { 1.2f, 2.3f, 4.5f };
    private static final String FIELD_NAME = "test_vector";

    private String description;
    private KNNQueryBuilder expected;
    private Integer k;
    private Map<String, ?> methodParameters;
    private Float maxDistance;
    private Float minScore;
    private RescoreContext rescoreContext;
    private String exactSearchSpaceType;

    @ParametersFactory(argumentFormatting = "description:%1$s; k:%3$s, efSearch:%4$s, maxDist:%5$s, minScore:%6$s, rescoreContext:%6$s")
    public static Collection<Object[]> validParameters() {
        return Arrays.asList(
            $$(
                $(
                    "valid knn with k",
                    KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).k(10).build(),
                    10,
                    null,
                    null,
                    null,
                    null,
                    null
                ),
                $(
                    "valid knn with k and efSearch",
                    KNNQueryBuilder.builder()
                        .fieldName(FIELD_NAME)
                        .vector(QUERY_VECTOR)
                        .k(10)
                        .methodParameters(Map.of("ef_search", 12))
                        .build(),
                    10,
                    Map.of("ef_search", 12),
                    null,
                    null,
                    null,
                    null
                ),
                $(
                    "valid knn with maxDis",
                    KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).maxDistance(10.0f).build(),
                    null,
                    null,
                    10.0f,
                    null,
                    null,
                    null
                ),
                $(
                    "valid knn with minScore",
                    KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).minScore(10.0f).build(),
                    null,
                    null,
                    null,
                    10.0f,
                    null,
                    null
                ),
                $(
                    "valid knn with rescore",
                    KNNQueryBuilder.builder()
                        .fieldName(FIELD_NAME)
                        .vector(QUERY_VECTOR)
                        .rescoreContext(RescoreContext.getDefault())
                        .minScore(10.0f)
                        .build(),
                    null,
                    null,
                    null,
                    10.0f,
                    RescoreContext.getDefault(),
                    null
                ),
                $(
                    "valid knn with exactSearchSpaceType",
                    KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).k(2).exactSearchSpaceType("cosinesimil").build(),
                    2,
                    null,
                    null,
                    null,
                    null,
                    "cosinesimil"
                )
            )
        );
    }

    public void testValidBuilder() {
        assertEquals(
            expected,
            KNNQueryBuilder.builder()
                .fieldName(FIELD_NAME)
                .vector(QUERY_VECTOR)
                .k(k)
                .methodParameters(methodParameters)
                .maxDistance(maxDistance)
                .minScore(minScore)
                .rescoreContext(rescoreContext)
                .exactSearchSpaceType(exactSearchSpaceType)
                .build()
        );
    }
}

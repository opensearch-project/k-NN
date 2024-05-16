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
import org.opensearch.knn.KNNTestCase;

import java.util.Arrays;
import java.util.Collection;

import static com.carrotsearch.randomizedtesting.RandomizedTest.$;
import static com.carrotsearch.randomizedtesting.RandomizedTest.$$;

@AllArgsConstructor
public class KNNQueryBuilderValidParamsTests extends KNNTestCase {

    private static final float[] QUERY_VECTOR = new float[] { 1.2f, 2.3f, 4.5f };
    private static final String FIELD_NAME = "test_vector";

    private String description;
    private KNNQueryBuilder expected;
    private Integer k;
    private Integer efSearch;
    private Float maxDistance;
    private Float minScore;

    @ParametersFactory(argumentFormatting = "description:%1$s; k:%3$s, efSearch:%4$s, maxDist:%5$s, minScore:%6$s")
    public static Collection<Object[]> validParameters() {
        return Arrays.asList(
            $$(
                $(
                    "valid knn with k",
                    KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).k(10).build(),
                    10,
                    null,
                    null,
                    null
                ),
                $(
                    "valid knn with k and efSearch",
                    KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).k(10).efSearch(12).build(),
                    10,
                    12,
                    null,
                    null
                ),
                $(
                    "valid knn with maxDis",
                    KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).maxDistance(10.0f).build(),
                    null,
                    null,
                    10.0f,
                    null
                ),
                $(
                    "valid knn with minScore",
                    KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).minScore(10.0f).build(),
                    null,
                    null,
                    null,
                    10.0f
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
                .efSearch(efSearch)
                .maxDistance(maxDistance)
                .minScore(minScore)
                .build()
        );
    }
}

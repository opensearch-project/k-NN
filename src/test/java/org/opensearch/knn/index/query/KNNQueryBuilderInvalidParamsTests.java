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
public class KNNQueryBuilderInvalidParamsTests extends KNNTestCase {

    private static final float[] QUERY_VECTOR = new float[] { 1.2f, 2.3f, 4.5f };
    private static final String FIELD_NAME = "test_vector";

    private String description;
    private String expectedMessage;
    private KNNQueryBuilder.Builder knnQueryBuilderBuilder;

    @ParametersFactory(argumentFormatting = "description:%1$s; expectedMessage:%2$s; querybuilder:%3$s")
    public static Collection<Object[]> invalidParameters() {
        return Arrays.asList(
            $$(
                $("fieldName absent", "[knn] requires fieldName", KNNQueryBuilder.builder().k(1).vector(QUERY_VECTOR)),
                $("vector absent", "[knn] requires query vector", KNNQueryBuilder.builder().k(1).fieldName(FIELD_NAME)),
                $(
                    "vector empty",
                    "[knn] query vector is empty",
                    KNNQueryBuilder.builder().k(1).fieldName(FIELD_NAME).vector(new float[] {})
                ),
                $(
                    "Neither knn nor radial search",
                    "[knn] requires exactly one of k, distance or score to be set",
                    KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR)
                ),
                $(
                    "max distance and k present",
                    "[knn] requires exactly one of k, distance or score to be set",
                    KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).k(1).maxDistance(10f)
                ),
                $(
                    "min_score and k present",
                    "[knn] requires exactly one of k, distance or score to be set",
                    KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).k(1).minScore(1.0f)
                ),
                $(
                    "max_dist and min_score present",
                    "[knn] requires exactly one of k, distance or score to be set",
                    KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).maxDistance(1.0f).minScore(1.0f)
                ),
                $(
                    "max_dist, k and min_score present",
                    "[knn] requires exactly one of k, distance or score to be set",
                    KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).k(1).maxDistance(1.0f).minScore(1.0f)
                ),
                $(
                    "-ve k value",
                    "[knn] requires k to be in the range (0, 10000]",
                    KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).k(-1)
                ),
                $(
                    "k value greater than max",
                    "[knn] requires k to be in the range (0, 10000]",
                    KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).k(10001)
                ),
                $(
                    "efSearch 0",
                    "[knn] requires ef_Search greater than 0",
                    KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).efSearch(0).k(10)
                ),
                $(
                    "efSearch -ve",
                    "[knn] requires ef_Search greater than 0",
                    KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).efSearch(-1).k(10)
                ),
                $(
                    "efSearch for radial search min score",
                    "[knn] ef_Search is currently not supported for radial search",
                    KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).efSearch(2).minScore(1.0f)
                ),
                $(
                    "efSearch for radial search max dist",
                    "[knn] ef_Search is currently not supported for radial search",
                    KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).efSearch(2).maxDistance(1.0f)
                ),

                $(
                    "min score less than 0",
                    "[knn] requires minScore to be greater than 0",
                    KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).minScore(-1f)
                )
            )
        );
    }

    public void testInvalidBuilder() {
        Throwable exception = expectThrows(IllegalArgumentException.class, () -> knnQueryBuilderBuilder.build());
        assertEquals(expectedMessage, exception.getMessage(), expectedMessage);
    }
}
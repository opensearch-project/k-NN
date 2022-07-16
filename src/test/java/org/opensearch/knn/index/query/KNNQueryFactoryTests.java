/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.KnnVectorQuery;
import org.apache.lucene.search.Query;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.util.KNNEngine;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class KNNQueryFactoryTests extends KNNTestCase {
    private final int testQueryDimension = 17;
    private final float[] testQueryVector = new float[testQueryDimension];
    private final String testIndexName = "test-index";
    private final String testFieldName = "test-field";
    private final int testK = 10;

    public void testCreateCustomKNNQuery() {
        for (KNNEngine knnEngine : KNNEngine.getEnginesThatCreateCustomSegmentFiles()) {
            Query query = KNNQueryFactory.create(knnEngine, testIndexName, testFieldName, testQueryVector, testK);
            assertTrue(query instanceof CustomKNNQuery);

            assertEquals(testIndexName, ((CustomKNNQuery) query).getIndexName());
            assertEquals(testFieldName, ((CustomKNNQuery) query).getField());
            assertEquals(testQueryVector, ((CustomKNNQuery) query).getQueryVector());
            assertEquals(testK, ((CustomKNNQuery) query).getK());
        }
    }

    public void testCreateLuceneDefaultQuery() {
        List<KNNEngine> luceneDefaultQueryEngineList = Arrays.stream(KNNEngine.values())
            .filter(knnEngine -> !KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine))
            .collect(Collectors.toList());
        for (KNNEngine knnEngine : luceneDefaultQueryEngineList) {
            Query query = KNNQueryFactory.create(knnEngine, testIndexName, testFieldName, testQueryVector, testK);
            assertTrue(query instanceof KnnVectorQuery);
        }
    }
}

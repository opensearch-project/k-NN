/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.TopDocs;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Arrays;
import java.util.stream.Collectors;

public class TopApproxKnnCollectorTests extends OpenSearchTestCase {

    public void testCollect_thenSuccess() {
        float similarity = randomFloat();
        KNNEngine knnEngine = randomFrom(KNNEngine.FAISS, KNNEngine.LUCENE);
        SpaceType spaceType = randomFrom(
            Arrays.stream(SpaceType.values()).filter(space -> space != SpaceType.UNDEFINED).collect(Collectors.toList())
        );
        TopApproxKnnCollector collector = new TopApproxKnnCollector(1, knnEngine, spaceType);
        assertTrue(collector.collect(1, similarity));
        TopDocs topDocs = collector.topDocs();
        assertEquals(1, topDocs.scoreDocs.length);
        assertEquals(knnEngine.score(similarity, spaceType), topDocs.scoreDocs[0].score, 0.01);
    }

    public void testCollect_MoreThanK_thenReturnTopKResults() {
        TopApproxKnnCollector collector = new TopApproxKnnCollector(1, KNNEngine.FAISS, SpaceType.INNER_PRODUCT);
        assertTrue(collector.collect(1, 0.5f));
        assertTrue(collector.collect(2, 0.7f));
        TopDocs topDocs = collector.topDocs();
        assertEquals(1, topDocs.scoreDocs.length);
        assertEquals(KNNEngine.FAISS.score(0.7f, SpaceType.INNER_PRODUCT), topDocs.scoreDocs[0].score, 0.01);
    }

}

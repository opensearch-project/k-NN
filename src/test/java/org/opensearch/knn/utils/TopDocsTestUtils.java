/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.utils;

import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;

import java.util.HashMap;
import java.util.Map;

public final class TopDocsTestUtils {
    public static TopDocs buildTopDocs(Map<Integer, Float> result) {
        ScoreDoc[] allScoreDocs = result.entrySet()
            .stream()
            .map(entry -> new ScoreDoc(entry.getKey(), entry.getValue()))
            .toArray(ScoreDoc[]::new);

        return new TopDocs(new TotalHits(result.size(), TotalHits.Relation.EQUAL_TO), allScoreDocs);
    }

    public static Map<Integer, Float> convertTopDocsToMap(TopDocs topDocs) {

        Map<Integer, Float> resultMap = new HashMap<>();
        for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
            resultMap.put(scoreDoc.doc, scoreDoc.score);
        }
        return resultMap;
    }

}

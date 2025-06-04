/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.util.DocIdSetBuilder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

/**
 * Utility class used for processing results
 */
public final class ResultUtil {

    /**
     * Reduce the results to only include the top k results across all leaf results
     *
     * @param perLeafResults Results from the list
     * @param k the number of results across all leaf results to return
     */
    public static void reduceToTopK(final List<PerLeafResult> perLeafResults, final int k) {
        // Iterate over all scores to get min competitive score
        PriorityQueue<Float> topKMinQueue = new PriorityQueue<>(k);

        int count = 0;
        for (PerLeafResult perLeafResult : perLeafResults) {
            count += perLeafResult.getResult().scoreDocs.length;
            for (ScoreDoc scoreDoc : perLeafResult.getResult().scoreDocs) {
                if (topKMinQueue.size() < k) {
                    topKMinQueue.add(scoreDoc.score);
                } else if (topKMinQueue.peek() != null && scoreDoc.score > topKMinQueue.peek()) {
                    topKMinQueue.poll();
                    topKMinQueue.add(scoreDoc.score);
                }
            }
        }

        // If there are at most k results across everything, then no need to filter anything out
        if (count <= k) {
            return;
        }

        // Reduce the results based on min competitive score
        float minScore = topKMinQueue.peek() == null ? -Float.MAX_VALUE : topKMinQueue.peek();
        perLeafResults.forEach(results -> {
            List<ScoreDoc> filteredScoreDocList = new ArrayList<>();
            for (ScoreDoc scoreDoc : results.getResult().scoreDocs) {
                if (scoreDoc.score >= minScore) {
                    filteredScoreDocList.add(scoreDoc);
                }
            }
            ScoreDoc[] filteredScoreDoc = filteredScoreDocList.toArray(new ScoreDoc[0]);
            TotalHits totalHits = new TotalHits(filteredScoreDoc.length, TotalHits.Relation.EQUAL_TO);
            results.setResult(new TopDocs(totalHits, filteredScoreDoc));
        });
    }

    /**
     * Convert map of docs to doc id set iterator
     *
     * @param resultMap Map of results
     * @return Doc id set iterator
     * @throws IOException If an error occurs during the search.
     */
    public static DocIdSetIterator resultMapToDocIds(Map<Integer, Float> resultMap) throws IOException {
        if (resultMap.isEmpty()) {
            return DocIdSetIterator.empty();
        }
        final int maxDoc = Collections.max(resultMap.keySet()) + 1;
        return resultMapToDocIds(resultMap, maxDoc);
    }

    /**
     * Convert map of docs to doc id set iterator
     *
     * @param resultMap Map of results
     * @param maxDoc Max doc id
     * @return Doc id set iterator
     * @throws IOException If an error occurs during the search.
     */
    public static DocIdSetIterator resultMapToDocIds(Map<Integer, Float> resultMap, final int maxDoc) throws IOException {
        if (resultMap.isEmpty()) {
            return DocIdSetIterator.empty();
        }
        final DocIdSetBuilder docIdSetBuilder = new DocIdSetBuilder(maxDoc);
        final DocIdSetBuilder.BulkAdder setAdder = docIdSetBuilder.grow(resultMap.size());
        for (int doc : resultMap.keySet()) {
            setAdder.add(doc);
        }
        return docIdSetBuilder.build().iterator();
    }

    /**
     * COnvert map of results to top docs. Doc ids have proper offset
     *
     * @param resultMap map of scores for the leafs
     * @param segmentOffset Offset to apply to ids to make them shard ids
     * @return Top docs
     */
    public static TopDocs resultMapToTopDocs(Map<Integer, Float> resultMap, int segmentOffset) {
        if (resultMap.isEmpty()) {
            return new TopDocs(new TotalHits(0, TotalHits.Relation.EQUAL_TO), new ScoreDoc[0]);
        }

        int totalHits = 0;
        final List<ScoreDoc> scoreDocs = new ArrayList<>();
        final List<Map.Entry<Integer, Float>> topScores = new ArrayList<>(resultMap.entrySet());
        topScores.sort(Map.Entry.<Integer, Float>comparingByValue().reversed());
        for (Map.Entry<Integer, Float> entry : topScores) {
            ScoreDoc scoreDoc = new ScoreDoc(entry.getKey() + segmentOffset, entry.getValue());
            scoreDocs.add(scoreDoc);
            totalHits++;
        }

        return new TopDocs(new TotalHits(totalHits, TotalHits.Relation.EQUAL_TO), scoreDocs.toArray(ScoreDoc[]::new));
    }
}

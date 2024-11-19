/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.util.BitSet;
import org.junit.Assert;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class ResultUtilTests extends KNNTestCase {

    public void testReduceToTopK() {
        int firstPassK = 20;
        int finalK = 10;
        int segmentCount = 5;

        List<Map<Integer, Float>> initialLeafResults = getRandomListOfResults(firstPassK, segmentCount);
        List<Map<Integer, Float>> reducedLeafResults = initialLeafResults.stream().map(HashMap::new).collect(Collectors.toList());
        ResultUtil.reduceToTopK(reducedLeafResults, finalK);
        assertTopK(initialLeafResults, reducedLeafResults, finalK);

        firstPassK = 5;
        finalK = 20;
        segmentCount = 1;

        initialLeafResults = getRandomListOfResults(firstPassK, segmentCount);
        reducedLeafResults = initialLeafResults.stream().map(HashMap::new).collect(Collectors.toList());
        ResultUtil.reduceToTopK(reducedLeafResults, finalK);
        assertTopK(initialLeafResults, reducedLeafResults, firstPassK);
    }

    public void testResultMapToMatchBitSet() throws IOException {
        int firstPassK = 35;
        Map<Integer, Float> perLeafResults = getRandomResults(firstPassK);
        BitSet resultBitset = ResultUtil.resultMapToMatchBitSet(perLeafResults);
        assertResultMapToMatchBitSet(perLeafResults, resultBitset);
    }

    public void testResultMapToMatchBitSet_whenResultMapEmpty_thenReturnEmptyOptional() throws IOException {
        BitSet resultBitset = ResultUtil.resultMapToMatchBitSet(Collections.emptyMap());
        Assert.assertNull(resultBitset);

        BitSet resultBitset2 = ResultUtil.resultMapToMatchBitSet(null);
        Assert.assertNull(resultBitset2);
    }

    public void testResultMapToDocIds() throws IOException {
        int firstPassK = 42;
        Map<Integer, Float> perLeafResults = getRandomResults(firstPassK);
        final int maxDoc = Collections.max(perLeafResults.keySet()) + 1;
        DocIdSetIterator resultDocIdSetIterator = ResultUtil.resultMapToDocIds(perLeafResults, maxDoc);
        assertResultMapToDocIdSetIterator(perLeafResults, resultDocIdSetIterator);
    }

    public void testResultMapToTopDocs() {
        int k = 18;
        int offset = 121;
        Map<Integer, Float> perLeafResults = getRandomResults(k);
        TopDocs topDocs = ResultUtil.resultMapToTopDocs(perLeafResults, offset);
        assertResultMapToTopDocs(perLeafResults, topDocs, k, offset);
    }

    private void assertResultMapToTopDocs(Map<Integer, Float> perLeafResults, TopDocs topDocs, int k, int offset) {
        assertEquals(k, topDocs.totalHits.value);
        float previousScore = Float.MAX_VALUE;
        for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
            assertTrue(perLeafResults.containsKey(scoreDoc.doc - offset));
            assertEquals(perLeafResults.get(scoreDoc.doc - offset), scoreDoc.score, 0.0001);
            assertTrue(previousScore > scoreDoc.score);
            previousScore = scoreDoc.score;
        }
    }

    private void assertTopK(List<Map<Integer, Float>> beforeResults, List<Map<Integer, Float>> reducedResults, int expectedK) {
        assertEquals(beforeResults.size(), reducedResults.size());
        assertEquals(expectedK, reducedResults.stream().map(Map::size).reduce(Integer::sum).orElse(-1).intValue());
        float minScore = getMinScore(reducedResults);
        int count = 0;
        for (Map<Integer, Float> result : beforeResults) {
            for (float score : result.values()) {
                if (score >= minScore) {
                    count++;
                }
            }
        }
        assertEquals(expectedK, count);
    }

    private void assertResultMapToMatchBitSet(Map<Integer, Float> resultsMap, BitSet resultBitset) {
        assertEquals(resultsMap.size(), resultBitset.cardinality());
        for (Integer docId : resultsMap.keySet()) {
            assertTrue(resultBitset.get(docId));
        }
    }

    private void assertResultMapToDocIdSetIterator(Map<Integer, Float> resultsMap, DocIdSetIterator resultDocIdSetIterator)
        throws IOException {
        int count = 0;
        int docId = resultDocIdSetIterator.nextDoc();
        while (docId != DocIdSetIterator.NO_MORE_DOCS) {
            assertTrue(resultsMap.containsKey(docId));
            count++;
            docId = resultDocIdSetIterator.nextDoc();
        }
        assertEquals(resultsMap.size(), count);
    }

    private List<Map<Integer, Float>> getRandomListOfResults(int k, int segments) {
        List<Map<Integer, Float>> perLeafResults = new ArrayList<>();
        for (int i = 0; i < segments; i++) {
            perLeafResults.add(getRandomResults(k));
        }
        return perLeafResults;
    }

    private Map<Integer, Float> getRandomResults(int k) {
        Map<Integer, Float> results = new HashMap<>();
        for (int i = 0; i < k; i++) {
            results.put(i, random().nextFloat());
        }

        return results;
    }

    private float getMinScore(List<Map<Integer, Float>> perLeafResults) {
        float minScore = Float.MAX_VALUE;
        for (Map<Integer, Float> result : perLeafResults) {
            for (float score : result.values()) {
                if (score < minScore) {
                    minScore = score;
                }
            }
        }
        return minScore;
    }
}

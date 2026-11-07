/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import junit.framework.TestCase;
import org.apache.lucene.search.KnnByteVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;

import static org.mockito.Mockito.mock;

public class OSKnnByteVectorQueryTests extends TestCase {

    public void testConstructor() {
        String fieldName = "test_field";
        byte[] queryVector = { 1, 2, 3 };
        int luceneK = 10;
        int k = 5;
        Query filterQuery = mock(Query.class);

        OSKnnByteVectorQuery query = new OSKnnByteVectorQuery(fieldName, queryVector, luceneK, filterQuery, k);

        assertTrue(query instanceof KnnByteVectorQuery);
    }

    public void testMergeLeafResults() {
        String fieldName = "test_field";
        byte[] queryVector = { 1, 2, 3 };
        int luceneK = 10;
        int k = 3;
        Query filterQuery = mock(Query.class);

        OSKnnByteVectorQuery query = new OSKnnByteVectorQuery(fieldName, queryVector, luceneK, filterQuery, k);

        ScoreDoc[] scoreDocs1 = { new ScoreDoc(1, 0.9f), new ScoreDoc(2, 0.8f) };
        ScoreDoc[] scoreDocs2 = { new ScoreDoc(3, 0.7f), new ScoreDoc(4, 0.6f) };

        TopDocs topDocs1 = new TopDocs(new TotalHits(2, TotalHits.Relation.EQUAL_TO), scoreDocs1);
        TopDocs topDocs2 = new TopDocs(new TotalHits(2, TotalHits.Relation.EQUAL_TO), scoreDocs2);

        TopDocs[] perLeafResults = { topDocs1, topDocs2 };

        TopDocs result = query.mergeLeafResults(perLeafResults);

        assertEquals(k, result.scoreDocs.length);
        assertTrue(result.scoreDocs[0].score >= result.scoreDocs[1].score);
        assertTrue(result.scoreDocs[1].score >= result.scoreDocs[2].score);
    }

    public void testMergeLeafResults_withFewerResultsThanK() {
        String fieldName = "test_field";
        byte[] queryVector = { 1, 2, 3 };
        int luceneK = 10;
        int k = 5;
        Query filterQuery = mock(Query.class);

        OSKnnByteVectorQuery query = new OSKnnByteVectorQuery(fieldName, queryVector, luceneK, filterQuery, k);

        ScoreDoc[] scoreDocs = { new ScoreDoc(1, 0.9f), new ScoreDoc(2, 0.8f) };
        TopDocs topDocs = new TopDocs(new TotalHits(2, TotalHits.Relation.EQUAL_TO), scoreDocs);
        TopDocs[] perLeafResults = { topDocs };

        TopDocs result = query.mergeLeafResults(perLeafResults);

        assertEquals(2, result.scoreDocs.length);
    }

    public void testMergeLeafResultsWithRescore() {
        String fieldName = "test_field";
        byte[] queryVector = { 1, 2, 3 };
        int luceneK = 6;
        int k = 2;
        int rescoreK = 4;
        Query filterQuery = mock(Query.class);

        OSKnnByteVectorQuery query = new OSKnnByteVectorQuery(fieldName, queryVector, luceneK, filterQuery, k, rescoreK);

        ScoreDoc[] scoreDocs1 = { new ScoreDoc(1, 0.9f), new ScoreDoc(2, 0.8f), new ScoreDoc(5, 0.5f) };
        ScoreDoc[] scoreDocs2 = { new ScoreDoc(3, 0.7f), new ScoreDoc(4, 0.6f), new ScoreDoc(6, 0.4f) };

        TopDocs topDocs1 = new TopDocs(new TotalHits(3, TotalHits.Relation.EQUAL_TO), scoreDocs1);
        TopDocs topDocs2 = new TopDocs(new TotalHits(3, TotalHits.Relation.EQUAL_TO), scoreDocs2);

        TopDocs[] perLeafResults = { topDocs1, topDocs2 };

        TopDocs result = query.mergeLeafResults(perLeafResults);

        // When rescore is enabled, results are trimmed to rescoreK (not k, not luceneK)
        assertEquals(rescoreK, result.scoreDocs.length);
    }
}

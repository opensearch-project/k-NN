/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import junit.framework.TestCase;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;

import static org.mockito.Mockito.mock;

public class OSKnnFloatVectorQueryTests extends TestCase {

    public void testConstructor() {
        String fieldName = "test_field";
        float[] queryVector = { 1.0f, 2.0f, 3.0f };
        int luceneK = 10;
        int k = 5;
        boolean needsRescore = false;
        Query filterQuery = mock(Query.class);

        OSKnnFloatVectorQuery query = new OSKnnFloatVectorQuery(fieldName, queryVector, luceneK, filterQuery, k, needsRescore);

        assertTrue(query instanceof KnnFloatVectorQuery);
    }

    public void testMergeLeafResultsWithRescoreFalse() {
        String fieldName = "test_field";
        float[] queryVector = { 1.0f, 2.0f, 3.0f };
        int luceneK = 10;
        int k = 3;
        Query filterQuery = mock(Query.class);

        OSKnnFloatVectorQuery query = new OSKnnFloatVectorQuery(fieldName, queryVector, luceneK, filterQuery, k, false);

        // Create mock TopDocs with more results than k
        ScoreDoc[] scoreDocs1 = { new ScoreDoc(1, 0.9f), new ScoreDoc(2, 0.8f) };
        ScoreDoc[] scoreDocs2 = { new ScoreDoc(3, 0.7f), new ScoreDoc(4, 0.6f) };

        TopDocs topDocs1 = new TopDocs(new TotalHits(2, TotalHits.Relation.EQUAL_TO), scoreDocs1);
        TopDocs topDocs2 = new TopDocs(new TotalHits(2, TotalHits.Relation.EQUAL_TO), scoreDocs2);

        TopDocs[] perLeafResults = { topDocs1, topDocs2 };

        TopDocs result = query.mergeLeafResults(perLeafResults);

        // Should return only top k results
        assertEquals(k, result.scoreDocs.length);
        // Results should be sorted by score (highest first)
        assertTrue(result.scoreDocs[0].score >= result.scoreDocs[1].score);
        assertTrue(result.scoreDocs[1].score >= result.scoreDocs[2].score);
    }

    public void testMergeLeafResults_withFewerResultsThanK() {
        String fieldName = "test_field";
        float[] queryVector = { 1.0f, 2.0f, 3.0f };
        int luceneK = 10;
        int k = 5;
        Query filterQuery = mock(Query.class);

        OSKnnFloatVectorQuery query = new OSKnnFloatVectorQuery(fieldName, queryVector, luceneK, filterQuery, k, false);

        // Create mock TopDocs with fewer results than k
        ScoreDoc[] scoreDocs = { new ScoreDoc(1, 0.9f), new ScoreDoc(2, 0.8f) };
        TopDocs topDocs = new TopDocs(new TotalHits(2, TotalHits.Relation.EQUAL_TO), scoreDocs);
        TopDocs[] perLeafResults = { topDocs };

        TopDocs result = query.mergeLeafResults(perLeafResults);

        // Should return all available results (less than k)
        assertEquals(2, result.scoreDocs.length);
    }

    public void testMergeLeafResultsWithRescoreTrue() {
        String fieldName = "test_field";
        float[] queryVector = { 1.0f, 2.0f, 3.0f };
        int luceneK = 4;
        int k = 2;
        Query filterQuery = mock(Query.class);

        OSKnnFloatVectorQuery query = new OSKnnFloatVectorQuery(fieldName, queryVector, luceneK, filterQuery, k, true);

        ScoreDoc[] scoreDocs1 = { new ScoreDoc(1, 0.9f), new ScoreDoc(2, 0.8f) };
        ScoreDoc[] scoreDocs2 = { new ScoreDoc(3, 0.7f), new ScoreDoc(4, 0.6f) };

        TopDocs topDocs1 = new TopDocs(new TotalHits(2, TotalHits.Relation.EQUAL_TO), scoreDocs1);
        TopDocs topDocs2 = new TopDocs(new TotalHits(2, TotalHits.Relation.EQUAL_TO), scoreDocs2);

        TopDocs[] perLeafResults = { topDocs1, topDocs2 };

        TopDocs result = query.mergeLeafResults(perLeafResults);

        // When needsRescore is true, results are not reduced to k and instead reduced to luceneK
        assertEquals(luceneK, result.scoreDocs.length);
    }
}

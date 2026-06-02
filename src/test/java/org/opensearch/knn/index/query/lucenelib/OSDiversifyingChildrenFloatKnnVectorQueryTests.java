/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import junit.framework.TestCase;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.DiversifyingChildrenFloatKnnVectorQuery;

import static org.mockito.Mockito.mock;

public class OSDiversifyingChildrenFloatKnnVectorQueryTests extends TestCase {

    public void testConstructor() {
        String fieldName = "test_field";
        float[] queryVector = { 1.0f, 2.0f, 3.0f };
        int luceneK = 10;
        int k = 5;
        int rescoreK = 0;
        boolean expandNestedDocs = false;
        Query filterQuery = mock(Query.class);
        BitSetProducer parentFilter = mock(BitSetProducer.class);

        OSDiversifyingChildrenFloatKnnVectorQuery query = new OSDiversifyingChildrenFloatKnnVectorQuery(
            fieldName,
            queryVector,
            filterQuery,
            luceneK,
            parentFilter,
            k,
            rescoreK,
            expandNestedDocs
        );

        assertTrue(query instanceof DiversifyingChildrenFloatKnnVectorQuery);
    }

    public void testMergeLeafResultsWithRescoreDisabled() {
        String fieldName = "test_field";
        float[] queryVector = { 1.0f, 2.0f, 3.0f };
        int luceneK = 10;
        int k = 3;
        int rescoreK = 0;
        boolean expandNestedDocs = false;
        Query filterQuery = mock(Query.class);
        BitSetProducer parentFilter = mock(BitSetProducer.class);

        OSDiversifyingChildrenFloatKnnVectorQuery query = new OSDiversifyingChildrenFloatKnnVectorQuery(
            fieldName,
            queryVector,
            filterQuery,
            luceneK,
            parentFilter,
            k,
            rescoreK,
            expandNestedDocs
        );

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
        float[] queryVector = { 1.0f, 2.0f, 3.0f };
        int luceneK = 10;
        int k = 5;
        int rescoreK = 0;
        boolean expandNestedDocs = false;
        Query filterQuery = mock(Query.class);
        BitSetProducer parentFilter = mock(BitSetProducer.class);

        OSDiversifyingChildrenFloatKnnVectorQuery query = new OSDiversifyingChildrenFloatKnnVectorQuery(
            fieldName,
            queryVector,
            filterQuery,
            luceneK,
            parentFilter,
            k,
            rescoreK,
            expandNestedDocs
        );

        ScoreDoc[] scoreDocs = { new ScoreDoc(1, 0.9f), new ScoreDoc(2, 0.8f) };
        TopDocs topDocs = new TopDocs(new TotalHits(2, TotalHits.Relation.EQUAL_TO), scoreDocs);
        TopDocs[] perLeafResults = { topDocs };

        TopDocs result = query.mergeLeafResults(perLeafResults);

        assertEquals(2, result.scoreDocs.length);
    }

    public void testMergeLeafResultsWithRescoreEnabled() {
        String fieldName = "test_field";
        float[] queryVector = { 1.0f, 2.0f, 3.0f };
        int luceneK = 10;
        int k = 3;
        int rescoreK = 6;
        boolean expandNestedDocs = false;
        Query filterQuery = mock(Query.class);
        BitSetProducer parentFilter = mock(BitSetProducer.class);

        OSDiversifyingChildrenFloatKnnVectorQuery query = new OSDiversifyingChildrenFloatKnnVectorQuery(
            fieldName,
            queryVector,
            filterQuery,
            luceneK,
            parentFilter,
            k,
            rescoreK,
            expandNestedDocs
        );

        ScoreDoc[] scoreDocs1 = { new ScoreDoc(1, 0.9f), new ScoreDoc(2, 0.8f), new ScoreDoc(5, 0.5f) };
        ScoreDoc[] scoreDocs2 = { new ScoreDoc(3, 0.7f), new ScoreDoc(4, 0.6f), new ScoreDoc(6, 0.4f) };

        TopDocs topDocs1 = new TopDocs(new TotalHits(3, TotalHits.Relation.EQUAL_TO), scoreDocs1);
        TopDocs topDocs2 = new TopDocs(new TotalHits(3, TotalHits.Relation.EQUAL_TO), scoreDocs2);

        TopDocs[] perLeafResults = { topDocs1, topDocs2 };

        TopDocs result = query.mergeLeafResults(perLeafResults);

        // When rescoreK > 0 and expandNestedDocs is false, should trim to rescoreK (not luceneK or k)
        assertEquals(rescoreK, result.scoreDocs.length);
    }

    public void testMergeLeafResultsWithExpandNestedDocs() {
        String fieldName = "test_field";
        float[] queryVector = { 1.0f, 2.0f, 3.0f };
        int luceneK = 10;
        int k = 3;
        int rescoreK = 0;
        boolean expandNestedDocs = true;
        Query filterQuery = mock(Query.class);
        BitSetProducer parentFilter = mock(BitSetProducer.class);

        OSDiversifyingChildrenFloatKnnVectorQuery query = new OSDiversifyingChildrenFloatKnnVectorQuery(
            fieldName,
            queryVector,
            filterQuery,
            luceneK,
            parentFilter,
            k,
            rescoreK,
            expandNestedDocs
        );

        ScoreDoc[] scoreDocs1 = { new ScoreDoc(1, 0.9f), new ScoreDoc(2, 0.8f) };
        ScoreDoc[] scoreDocs2 = { new ScoreDoc(3, 0.7f), new ScoreDoc(4, 0.6f) };

        TopDocs topDocs1 = new TopDocs(new TotalHits(2, TotalHits.Relation.EQUAL_TO), scoreDocs1);
        TopDocs topDocs2 = new TopDocs(new TotalHits(2, TotalHits.Relation.EQUAL_TO), scoreDocs2);

        TopDocs[] perLeafResults = { topDocs1, topDocs2 };

        TopDocs result = query.mergeLeafResults(perLeafResults);

        // When expandNestedDocs is true, should reduce to k regardless of rescoreK
        assertEquals(k, result.scoreDocs.length);
    }

    public void testConstructorWithoutExpandNestedDocs() {
        String fieldName = "test_field";
        float[] queryVector = { 1.0f, 2.0f, 3.0f };
        int luceneK = 10;
        int k = 3;
        int rescoreK = 0;
        Query filterQuery = mock(Query.class);
        BitSetProducer parentFilter = mock(BitSetProducer.class);

        // Test the overloaded constructor without expandNestedDocs parameter
        OSDiversifyingChildrenFloatKnnVectorQuery query = new OSDiversifyingChildrenFloatKnnVectorQuery(
            fieldName,
            queryVector,
            filterQuery,
            luceneK,
            parentFilter,
            k,
            rescoreK
        );

        assertTrue(query instanceof DiversifyingChildrenFloatKnnVectorQuery);

        // Verify it behaves as if expandNestedDocs is false
        ScoreDoc[] scoreDocs1 = { new ScoreDoc(1, 0.9f), new ScoreDoc(2, 0.8f) };
        ScoreDoc[] scoreDocs2 = { new ScoreDoc(3, 0.7f), new ScoreDoc(4, 0.6f) };

        TopDocs topDocs1 = new TopDocs(new TotalHits(2, TotalHits.Relation.EQUAL_TO), scoreDocs1);
        TopDocs topDocs2 = new TopDocs(new TotalHits(2, TotalHits.Relation.EQUAL_TO), scoreDocs2);

        TopDocs[] perLeafResults = { topDocs1, topDocs2 };
        TopDocs result = query.mergeLeafResults(perLeafResults);

        // Should reduce to k (default behavior when expandNestedDocs is false and rescoreK is 0)
        assertEquals(k, result.scoreDocs.length);
    }

    public void testMergeLeafResultsWithRescoreAndExpandNestedDocs() {
        String fieldName = "test_field";
        float[] queryVector = { 1.0f, 2.0f, 3.0f };
        int luceneK = 10;
        int k = 3;
        int rescoreK = 6;
        boolean expandNestedDocs = true;
        Query filterQuery = mock(Query.class);
        BitSetProducer parentFilter = mock(BitSetProducer.class);

        OSDiversifyingChildrenFloatKnnVectorQuery query = new OSDiversifyingChildrenFloatKnnVectorQuery(
            fieldName,
            queryVector,
            filterQuery,
            luceneK,
            parentFilter,
            k,
            rescoreK,
            expandNestedDocs
        );

        ScoreDoc[] scoreDocs1 = { new ScoreDoc(1, 0.9f), new ScoreDoc(2, 0.8f) };
        ScoreDoc[] scoreDocs2 = { new ScoreDoc(3, 0.7f), new ScoreDoc(4, 0.6f) };

        TopDocs topDocs1 = new TopDocs(new TotalHits(2, TotalHits.Relation.EQUAL_TO), scoreDocs1);
        TopDocs topDocs2 = new TopDocs(new TotalHits(2, TotalHits.Relation.EQUAL_TO), scoreDocs2);

        TopDocs[] perLeafResults = { topDocs1, topDocs2 };

        TopDocs result = query.mergeLeafResults(perLeafResults);

        // When expandNestedDocs is true, should reduce to k even if rescoreK > 0
        assertEquals(k, result.scoreDocs.length);
    }

    public void testMergeLeafResultsWithRescoreK_trimsToRescoreKNotLuceneK() {
        String fieldName = "test_field";
        float[] queryVector = { 1.0f, 2.0f, 3.0f };
        // luceneK=256 (ef_search dominated), rescoreK=200 (oversample dominated), k=100
        int luceneK = 256;
        int k = 100;
        int rescoreK = 200;
        boolean expandNestedDocs = false;
        Query filterQuery = mock(Query.class);
        BitSetProducer parentFilter = mock(BitSetProducer.class);

        OSDiversifyingChildrenFloatKnnVectorQuery query = new OSDiversifyingChildrenFloatKnnVectorQuery(
            fieldName,
            queryVector,
            filterQuery,
            luceneK,
            parentFilter,
            k,
            rescoreK,
            expandNestedDocs
        );

        // Create enough results to exceed rescoreK but be within luceneK
        ScoreDoc[] scoreDocs1 = new ScoreDoc[128];
        ScoreDoc[] scoreDocs2 = new ScoreDoc[128];
        for (int i = 0; i < 128; i++) {
            scoreDocs1[i] = new ScoreDoc(i, 1.0f - (i * 0.001f));
            scoreDocs2[i] = new ScoreDoc(128 + i, 0.5f - (i * 0.001f));
        }

        TopDocs topDocs1 = new TopDocs(new TotalHits(128, TotalHits.Relation.EQUAL_TO), scoreDocs1);
        TopDocs topDocs2 = new TopDocs(new TotalHits(128, TotalHits.Relation.EQUAL_TO), scoreDocs2);

        TopDocs[] perLeafResults = { topDocs1, topDocs2 };

        TopDocs result = query.mergeLeafResults(perLeafResults);

        // Should trim to rescoreK (200), not luceneK (256) and not k (100)
        assertEquals(rescoreK, result.scoreDocs.length);
    }
}

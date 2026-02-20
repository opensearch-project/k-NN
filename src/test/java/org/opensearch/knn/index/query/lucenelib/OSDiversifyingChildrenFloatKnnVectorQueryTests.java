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
        boolean needsRescore = false;
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
            needsRescore,
            expandNestedDocs
        );

        assertTrue(query instanceof DiversifyingChildrenFloatKnnVectorQuery);
    }

    public void testMergeLeafResultsWithRescoreDisabled() {
        String fieldName = "test_field";
        float[] queryVector = { 1.0f, 2.0f, 3.0f };
        int luceneK = 10;
        int k = 3;
        boolean needsRescore = false;
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
            needsRescore,
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
        boolean needsRescore = false;
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
            needsRescore,
            expandNestedDocs
        );

        ScoreDoc[] scoreDocs = { new ScoreDoc(1, 0.9f), new ScoreDoc(2, 0.8f) };
        TopDocs topDocs = new TopDocs(new TotalHits(2, TotalHits.Relation.EQUAL_TO), scoreDocs);
        TopDocs[] perLeafResults = { topDocs };

        TopDocs result = query.mergeLeafResults(perLeafResults);

        assertEquals(2, result.scoreDocs.length);
    }
}

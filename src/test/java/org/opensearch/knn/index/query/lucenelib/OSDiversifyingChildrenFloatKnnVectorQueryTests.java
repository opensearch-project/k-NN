/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import junit.framework.TestCase;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.DiversifyingChildrenFloatKnnVectorQuery;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.io.IOException;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class OSDiversifyingChildrenFloatKnnVectorQueryTests extends TestCase {

    private static final String FIELD_NAME = "test_field";
    private static final float[] QUERY_VECTOR = { 1.0f, 2.0f, 3.0f };

    private OSDiversifyingChildrenFloatKnnVectorQuery newQuery(int luceneK, int k, int rescoreK) {
        return new OSDiversifyingChildrenFloatKnnVectorQuery(
            FIELD_NAME,
            QUERY_VECTOR,
            mock(Query.class),
            luceneK,
            mock(BitSetProducer.class),
            k,
            rescoreK
        );
    }

    private static TopDocs[] twoLeavesOfTwo() {
        ScoreDoc[] scoreDocs1 = { new ScoreDoc(1, 0.9f), new ScoreDoc(2, 0.8f) };
        ScoreDoc[] scoreDocs2 = { new ScoreDoc(3, 0.7f), new ScoreDoc(4, 0.6f) };
        return new TopDocs[] {
            new TopDocs(new TotalHits(2, TotalHits.Relation.EQUAL_TO), scoreDocs1),
            new TopDocs(new TotalHits(2, TotalHits.Relation.EQUAL_TO), scoreDocs2) };
    }

    public void testConstructor() {
        OSDiversifyingChildrenFloatKnnVectorQuery query = newQuery(10, 5, RescoreContext.NO_RESCORE_NEEDED);

        assertTrue(query instanceof DiversifyingChildrenFloatKnnVectorQuery);
    }

    public void testMergeLeafResultsWithRescoreDisabled() {
        int k = 3;
        OSDiversifyingChildrenFloatKnnVectorQuery query = newQuery(10, k, RescoreContext.NO_RESCORE_NEEDED);

        TopDocs result = query.mergeLeafResults(twoLeavesOfTwo());

        assertEquals(k, result.scoreDocs.length);
        assertTrue(result.scoreDocs[0].score >= result.scoreDocs[1].score);
        assertTrue(result.scoreDocs[1].score >= result.scoreDocs[2].score);
    }

    public void testMergeLeafResults_withFewerResultsThanK() {
        OSDiversifyingChildrenFloatKnnVectorQuery query = newQuery(10, 5, RescoreContext.NO_RESCORE_NEEDED);

        ScoreDoc[] scoreDocs = { new ScoreDoc(1, 0.9f), new ScoreDoc(2, 0.8f) };
        TopDocs[] perLeafResults = { new TopDocs(new TotalHits(2, TotalHits.Relation.EQUAL_TO), scoreDocs) };

        TopDocs result = query.mergeLeafResults(perLeafResults);

        assertEquals(2, result.scoreDocs.length);
    }

    public void testMergeLeafResultsWithRescoreEnabled() {
        int rescoreK = 6;
        OSDiversifyingChildrenFloatKnnVectorQuery query = newQuery(10, 3, rescoreK);

        ScoreDoc[] scoreDocs1 = { new ScoreDoc(1, 0.9f), new ScoreDoc(2, 0.8f), new ScoreDoc(5, 0.5f) };
        ScoreDoc[] scoreDocs2 = { new ScoreDoc(3, 0.7f), new ScoreDoc(4, 0.6f), new ScoreDoc(6, 0.4f) };

        TopDocs[] perLeafResults = {
            new TopDocs(new TotalHits(3, TotalHits.Relation.EQUAL_TO), scoreDocs1),
            new TopDocs(new TotalHits(3, TotalHits.Relation.EQUAL_TO), scoreDocs2) };

        TopDocs result = query.mergeLeafResults(perLeafResults);

        // When rescoreK > 0, should trim to rescoreK (not luceneK or k)
        assertEquals(rescoreK, result.scoreDocs.length);
    }

    /**
     * Regression test for the expand_nested_docs + rescoring interaction. The candidate pool used to be cut
     * to k whenever expand_nested_docs was set, which left rescoring nothing to re-rank. The oversampled
     * pool now survives regardless, and ExpandNestedDocsQuery does the cut to k itself after rescoring.
     */
    public void testMergeLeafResults_whenRescoreEnabled_thenKeepsOversampledPoolForExpansion() {
        int k = 3;
        int rescoreK = 4;
        OSDiversifyingChildrenFloatKnnVectorQuery query = newQuery(10, k, rescoreK);

        TopDocs result = query.mergeLeafResults(twoLeavesOfTwo());

        assertEquals(rescoreK, result.scoreDocs.length);
    }

    public void testApproximateSearch_whenParentBitSetNull_thenReturnEmptyResults() throws IOException {
        BitSetProducer parentFilter = mock(BitSetProducer.class);
        LeafReaderContext context = mock(LeafReaderContext.class);
        KnnCollectorManager knnCollectorManager = mock(KnnCollectorManager.class);
        AcceptDocs acceptDocs = mock(AcceptDocs.class);

        when(parentFilter.getBitSet(context)).thenReturn(null);

        OSDiversifyingChildrenFloatKnnVectorQuery query = new OSDiversifyingChildrenFloatKnnVectorQuery(
            FIELD_NAME,
            QUERY_VECTOR,
            mock(Query.class),
            10,
            parentFilter,
            5,
            RescoreContext.NO_RESCORE_NEEDED
        );

        TopDocs result = query.approximateSearch(context, acceptDocs, Integer.MAX_VALUE, knnCollectorManager);

        assertNotNull(result);
        assertEquals(0, result.totalHits.value());
        assertEquals(0, result.scoreDocs.length);
    }

    public void testMergeLeafResultsWithRescoreK_trimsToRescoreKNotLuceneK() {
        // luceneK=256 (ef_search dominated), rescoreK=200 (oversample dominated), k=100
        int rescoreK = 200;
        OSDiversifyingChildrenFloatKnnVectorQuery query = newQuery(256, 100, rescoreK);

        // Create enough results to exceed rescoreK but be within luceneK
        ScoreDoc[] scoreDocs1 = new ScoreDoc[128];
        ScoreDoc[] scoreDocs2 = new ScoreDoc[128];
        for (int i = 0; i < 128; i++) {
            scoreDocs1[i] = new ScoreDoc(i, 1.0f - (i * 0.001f));
            scoreDocs2[i] = new ScoreDoc(128 + i, 0.5f - (i * 0.001f));
        }

        TopDocs[] perLeafResults = {
            new TopDocs(new TotalHits(128, TotalHits.Relation.EQUAL_TO), scoreDocs1),
            new TopDocs(new TotalHits(128, TotalHits.Relation.EQUAL_TO), scoreDocs2) };

        TopDocs result = query.mergeLeafResults(perLeafResults);

        // Should trim to rescoreK (200), not luceneK (256) and not k (100)
        assertEquals(rescoreK, result.scoreDocs.length);
    }
}

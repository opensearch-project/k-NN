/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.common;

import lombok.SneakyThrows;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexReaderContext;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.Weight;
import org.mockito.Mock;
import org.opensearch.test.OpenSearchTestCase;

import static org.mockito.Mockito.when;
import static org.mockito.MockitoAnnotations.openMocks;

public class DocAndScoreQueryTests extends OpenSearchTestCase {

    @Mock
    private LeafReaderContext leaf1;
    @Mock
    private IndexSearcher indexSearcher;
    @Mock
    private IndexReader reader;
    @Mock
    private IndexReaderContext readerContext;

    private DocAndScoreQuery objectUnderTest;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        openMocks(this);

        when(indexSearcher.getIndexReader()).thenReturn(reader);
        when(reader.getContext()).thenReturn(readerContext);
        when(readerContext.id()).thenReturn(1);
    }

    // Note: cannot test with multi leaf as there LeafReaderContext is readonly with no getters for some fields to mock
    public void testScorer() throws Exception {
        // Given
        int[] expectedDocs = { 0, 1, 2, 3, 4 };
        float[] expectedScores = { 0.1f, 1.2f, 2.3f, 5.1f, 3.4f };
        int[] findSegments = { 0, 2, 5 };
        objectUnderTest = new DocAndScoreQuery(4, expectedDocs, expectedScores, findSegments, 1);

        // When
        Scorer scorer1 = objectUnderTest.createWeight(indexSearcher, ScoreMode.COMPLETE, 1).scorer(leaf1);
        DocIdSetIterator iterator1 = scorer1.iterator();
        Scorer scorer2 = objectUnderTest.createWeight(indexSearcher, ScoreMode.COMPLETE, 1).scorer(leaf1);
        DocIdSetIterator iterator2 = scorer2.iterator();

        int[] actualDocs = new int[2];
        float[] actualScores = new float[2];
        int index = 0;
        while (iterator1.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
            actualDocs[index] = iterator1.docID();
            actualScores[index] = scorer1.score();
            ++index;
        }

        // Then
        assertEquals(2, iterator1.cost());
        assertArrayEquals(new int[] { 0, 1 }, actualDocs);
        assertArrayEquals(new float[] { 0.1f, 1.2f }, actualScores, 0.0001f);

        assertEquals(1.2f, scorer2.getMaxScore(1), 0.0001f);
        assertEquals(iterator2.advance(1), 1);
    }

    @SneakyThrows
    public void testWeight() {
        // Given
        int[] expectedDocs = { 0, 1, 2, 3, 4 };
        float[] expectedScores = { 0.1f, 1.2f, 2.3f, 5.1f, 3.4f };
        int[] findSegments = { 0, 2, 5 };
        Explanation expectedExplanation = Explanation.match(1.2f, "within top 4");

        // When
        objectUnderTest = new DocAndScoreQuery(4, expectedDocs, expectedScores, findSegments, 1);
        Weight weight = objectUnderTest.createWeight(indexSearcher, ScoreMode.COMPLETE, 1);
        Explanation explanation = weight.explain(leaf1, 1);

        // Then
        assertEquals(objectUnderTest, weight.getQuery());
        assertTrue(weight.isCacheable(leaf1));
        assertEquals(2, weight.count(leaf1));
        assertEquals(expectedExplanation, explanation);
        assertEquals(Explanation.noMatch("not in top 4"), weight.explain(leaf1, 9));
    }
}

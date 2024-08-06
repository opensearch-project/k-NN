/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.nativelib;

import lombok.SneakyThrows;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexReaderContext;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.util.Bits;
import org.mockito.ArgumentMatchers;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.opensearch.knn.index.query.KNNQuery;
import org.opensearch.knn.index.query.KNNWeight;
import org.opensearch.test.OpenSearchTestCase;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;

import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.mockito.MockitoAnnotations.openMocks;

public class NativeEngineKNNVectorQueryTests extends OpenSearchTestCase {

    @Mock
    private IndexSearcher searcher;
    @Mock
    private IndexReader reader;
    @Mock
    private KNNQuery knnQuery;
    @Mock
    private KNNWeight knnWeight;
    @Mock
    private TaskExecutor taskExecutor;
    @Mock
    private IndexReaderContext indexReaderContext;
    @Mock
    private LeafReaderContext leaf1;
    @Mock
    private LeafReaderContext leaf2;
    @Mock
    private LeafReader leafReader1;
    @Mock
    private LeafReader leafReader2;

    @InjectMocks
    private NativeEngineKnnVectorQuery objectUnderTest;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        openMocks(this);

        when(leaf1.reader()).thenReturn(leafReader1);
        when(leaf2.reader()).thenReturn(leafReader2);

        when(searcher.getIndexReader()).thenReturn(reader);
        when(knnQuery.createWeight(searcher, ScoreMode.COMPLETE, 1)).thenReturn(knnWeight);

        when(searcher.getTaskExecutor()).thenReturn(taskExecutor);
        when(taskExecutor.invokeAll(ArgumentMatchers.<Callable<TopDocs>>anyList())).thenAnswer(invocationOnMock -> {
            List<Callable<TopDocs>> callables = invocationOnMock.getArgument(0);
            List<TopDocs> topDocs = new ArrayList<>();
            for (Callable<TopDocs> callable : callables) {
                topDocs.add(callable.call());
            }
            return topDocs;
        });

        when(reader.getContext()).thenReturn(indexReaderContext);
    }

    @SneakyThrows
    public void testMultiLeaf() {
        // Given
        List<LeafReaderContext> leaves = List.of(leaf1, leaf2);
        when(reader.leaves()).thenReturn(leaves);

        when(knnWeight.searchLeaf(leaf1)).thenReturn(Map.of(0, 1.2f, 1, 5.1f, 2, 2.2f));
        when(knnWeight.searchLeaf(leaf2)).thenReturn(Map.of(4, 3.4f, 3, 5.1f));

        // Making sure there is deleted docs in one of the segments
        Bits liveDocs = mock(Bits.class);
        when(leafReader1.getLiveDocs()).thenReturn(liveDocs);
        when(leafReader2.getLiveDocs()).thenReturn(null);

        when(liveDocs.get(anyInt())).thenReturn(true);
        when(liveDocs.get(2)).thenReturn(false);
        when(liveDocs.get(1)).thenReturn(false);

        // k=4 to make sure we get topk results even if docs are deleted/less in one of the leaves
        when(knnQuery.getK()).thenReturn(4);

        when(indexReaderContext.id()).thenReturn(1);
        int[] expectedDocs = { 0, 3, 4 };
        float[] expectedScores = { 1.2f, 5.1f, 3.4f };
        int[] findSegments = { 0, 1, 3 };
        DocAndScoreQuery expected = new DocAndScoreQuery(4, expectedDocs, expectedScores, findSegments, 1);

        // When
        Query actual = objectUnderTest.rewrite(searcher);

        // Then
        assertEquals(expected, actual);
    }

    @SneakyThrows
    public void testSingleLeaf() {
        // Given
        List<LeafReaderContext> leaves = List.of(leaf1);
        when(reader.leaves()).thenReturn(leaves);
        when(knnWeight.searchLeaf(leaf1)).thenReturn(Map.of(0, 1.2f, 1, 5.1f, 2, 2.2f));
        when(knnQuery.getK()).thenReturn(4);

        when(indexReaderContext.id()).thenReturn(1);
        int[] expectedDocs = { 0, 1, 2 };
        float[] expectedScores = { 1.2f, 5.1f, 2.2f };
        int[] findSegments = { 0, 3 };
        DocAndScoreQuery expected = new DocAndScoreQuery(4, expectedDocs, expectedScores, findSegments, 1);

        // When
        Query actual = objectUnderTest.rewrite(searcher);

        // Then
        assertEquals(expected, actual);
    }

    @SneakyThrows
    public void testNoMatch() {
        // Given
        List<LeafReaderContext> leaves = List.of(leaf1);
        when(reader.leaves()).thenReturn(leaves);
        when(knnWeight.searchLeaf(leaf1)).thenReturn(Collections.emptyMap());
        when(knnQuery.getK()).thenReturn(4);
        // When
        Query actual = objectUnderTest.rewrite(searcher);

        // Then
        assertEquals(new MatchNoDocsQuery(), actual);
    }
}

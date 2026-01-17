/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CompositeReaderContext;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.junit.Before;
import org.junit.Test;
import org.mockito.MockedStatic;
import org.opensearch.lucene.ReentrantKnnCollectorManager;

import java.lang.reflect.Constructor;
import java.util.Collections;
import java.util.Map;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class ReentrantKnnCollectorManagerTests {
    private KnnCollectorManager delegateManager;
    private KnnCollector delegateCollector;
    private LeafReader reader;
    private LeafReaderContext ctx;
    private FloatVectorValues vectorValues;
    private VectorScorer scorer;
    private KnnSearchStrategy searchStrategy;
    private ReentrantKnnCollectorManager manager;

    @Before
    public void setUp() throws Exception {
        delegateManager = mock(KnnCollectorManager.class);
        delegateCollector = mock(KnnCollector.class);
        reader = mock(LeafReader.class);
        vectorValues = mock(FloatVectorValues.class);
        scorer = mock(VectorScorer.class);
        searchStrategy = mock(KnnSearchStrategy.class);

        // Mock final class LeafReaderContext
        ctx = mock(LeafReaderContext.class);
        when(ctx.reader()).thenReturn(reader);

        // Mock getFieldInfos so that FloatVectorValues.checkField() or scorer() doesn't throw
        FieldInfos fieldInfos = mock(FieldInfos.class);
        when(reader.getFieldInfos()).thenReturn(fieldInfos);

        // Stub the delegate collector
        when(delegateManager.newCollector(anyInt(), any(), any())).thenReturn(delegateCollector);

        // Seed TopDocs
        TopDocs seedTopDocs = new TopDocs(
            new TotalHits(2, TotalHits.Relation.EQUAL_TO),
            new ScoreDoc[] { new ScoreDoc(1, 0.9f), new ScoreDoc(2, 0.8f) }
        );

        manager = new ReentrantKnnCollectorManager(delegateManager, Map.of(0, seedTopDocs), new float[] { 1.0f, 2.0f }, "vector_field");
    }

    @Test
    public void testNormalCase_SeedsApplied() throws Exception {
        // Given: scorer returns a DocIndexIterator
        KnnVectorValues.DocIndexIterator docIndexIterator = mock(KnnVectorValues.DocIndexIterator.class);
        when(reader.getFloatVectorValues("vector_field")).thenReturn(vectorValues);
        when(vectorValues.scorer(any(float[].class))).thenReturn(scorer);
        when(scorer.iterator()).thenReturn(docIndexIterator);

        // When
        KnnCollector collector = manager.newCollector(10, searchStrategy, ctx);

        // Then
        assertNotNull("Collector should not be null", collector);
        verify(delegateManager).newCollector(eq(10), argThat(arg -> arg instanceof KnnSearchStrategy.Seeded), eq(ctx));
    }

    @Test
    public void testNullVectorValues_TriggersCheckField() throws Exception {
        when(reader.getFloatVectorValues("vector_field")).thenReturn(null);

        try (MockedStatic<FloatVectorValues> mocked = mockStatic(FloatVectorValues.class)) {
            manager.newCollector(10, searchStrategy, ctx);
            mocked.verify(() -> FloatVectorValues.checkField(reader, "vector_field"));
        }
    }

    @Test
    public void testByteVectorTypeNewCollector() {
        final byte[] query = new byte[100];
        doTestNewCollector(query);
    }

    @Test
    public void testFloatVectorTypeNewCollector() {
        final float[] query = new float[100];
        doTestNewCollector(query);
    }

    @SneakyThrows
    private void doTestNewCollector(final Object query) {
        final KnnCollectorManager knnCollectorManager = mock(KnnCollectorManager.class);
        final KnnCollector defaultDelegateCollector = mock(KnnCollector.class);
        final KnnCollector seededDelegateCollector = mock(KnnCollector.class);
        when(knnCollectorManager.newCollector(anyInt(), any(), any())).thenAnswer(invocation -> {
            final KnnSearchStrategy searchStrategy = invocation.getArgument(1);
            if (searchStrategy instanceof KnnSearchStrategy.Seeded) {
                return seededDelegateCollector;
            }

            return defaultDelegateCollector;
        });

        final int n = 100;
        final ScoreDoc[] scoreDocs = new ScoreDoc[n];
        for (int i = 0; i < n; i++) {
            scoreDocs[i] = new ScoreDoc(i, i);
        }
        final TopDocs topDocs = new TopDocs(new TotalHits(n, TotalHits.Relation.EQUAL_TO), scoreDocs);
        final Map<Integer, TopDocs> segmentOrdToResults = Collections.singletonMap(0, topDocs);
        final String field = "field";

        final ReentrantKnnCollectorManager reentrantKnnCollectorManager = new ReentrantKnnCollectorManager(
            knnCollectorManager,
            segmentOrdToResults,
            query,
            field
        );

        // LeafReaderContext(CompositeReaderContext parent, LeafReader reader, int ord, int docBase, int leafOrd, int leafDocBase)
        final Constructor<LeafReaderContext> ctor = LeafReaderContext.class.getDeclaredConstructor(
            CompositeReaderContext.class,
            LeafReader.class,
            int.class,
            int.class,
            int.class,
            int.class
        );
        ctor.setAccessible(true);
        final LeafReader leafReader = mock(LeafReader.class);
        if (query instanceof float[]) {
            final FloatVectorValues floatVectorValues = mock(FloatVectorValues.class);
            final VectorScorer scorer = mock(VectorScorer.class);
            final KnnVectorValues.DocIndexIterator docIndexIterator = mock(KnnVectorValues.DocIndexIterator.class);
            when(scorer.iterator()).thenReturn(docIndexIterator);
            when(floatVectorValues.scorer(any())).thenReturn(scorer);
            when(leafReader.getFloatVectorValues(field)).thenReturn(floatVectorValues);
        } else if (query instanceof byte[]) {
            final ByteVectorValues byteVectorValues = mock(ByteVectorValues.class);
            final VectorScorer scorer = mock(VectorScorer.class);
            final KnnVectorValues.DocIndexIterator docIndexIterator = mock(KnnVectorValues.DocIndexIterator.class);
            when(scorer.iterator()).thenReturn(docIndexIterator);
            when(byteVectorValues.scorer(any())).thenReturn(scorer);
            when(leafReader.getByteVectorValues(field)).thenReturn(byteVectorValues);
        }
        final LeafReaderContext leafReaderContext = ctor.newInstance(null, leafReader, 0, 0, 0, 0);
        final KnnCollector knnCollector = reentrantKnnCollectorManager.newCollector(
            10000,
            KnnSearchStrategy.Hnsw.DEFAULT,
            leafReaderContext
        );

        assertSame(seededDelegateCollector, knnCollector);
    }
}

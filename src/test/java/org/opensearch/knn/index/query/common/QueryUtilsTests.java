/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.common;

import junit.framework.TestCase;
import lombok.SneakyThrows;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.junit.Before;
import org.opensearch.knn.profile.query.KNNQueryTimingType;
import org.opensearch.search.profile.ContextualProfileBreakdown;
import org.opensearch.search.profile.Timer;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.times;

public class QueryUtilsTests extends TestCase {
    private Executor executor;
    private TaskExecutor taskExecutor;
    private QueryUtils queryUtils;

    @Before
    public void setUp() throws Exception {
        executor = Executors.newSingleThreadExecutor();
        taskExecutor = new TaskExecutor(executor);
        queryUtils = QueryUtils.getInstance();
    }

    @SneakyThrows
    public void testDoSearch_whenExecuted_thenSucceed() {
        IndexSearcher indexSearcher = mock(IndexSearcher.class);
        when(indexSearcher.getTaskExecutor()).thenReturn(taskExecutor);

        LeafReaderContext leafReaderContext1 = mock(LeafReaderContext.class);
        LeafReaderContext leafReaderContext2 = mock(LeafReaderContext.class);
        List<LeafReaderContext> leafReaderContexts = Arrays.asList(leafReaderContext1, leafReaderContext2);

        DocIdSetIterator docIdSetIterator = mock(DocIdSetIterator.class);
        when(docIdSetIterator.docID()).thenReturn(0, 1, DocIdSetIterator.NO_MORE_DOCS);
        Scorer scorer = mock(Scorer.class);
        when(scorer.iterator()).thenReturn(docIdSetIterator);
        when(scorer.docID()).thenReturn(0, 1, DocIdSetIterator.NO_MORE_DOCS);
        when(scorer.score()).thenReturn(10.f, 11.f, -1f);

        Weight weight = mock(Weight.class);
        when(weight.scorer(leafReaderContext1)).thenReturn(null);
        when(weight.scorer(leafReaderContext2)).thenReturn(scorer);

        // Run
        List<Map<Integer, Float>> results = queryUtils.doSearch(indexSearcher, leafReaderContexts, weight);

        // Verify
        assertEquals(2, results.size());
        assertEquals(0, results.get(0).size());
        assertEquals(2, results.get(1).size());
        assertEquals(10.f, results.get(1).get(0));
        assertEquals(11.f, results.get(1).get(1));

    }

    @SneakyThrows
    public void testDoSearchWithProfile_whenExecuted_thenSucceed() {
        IndexSearcher indexSearcher = mock(IndexSearcher.class);
        when(indexSearcher.getTaskExecutor()).thenReturn(taskExecutor);

        LeafReaderContext leafReaderContext1 = mock(LeafReaderContext.class);
        LeafReaderContext leafReaderContext2 = mock(LeafReaderContext.class);
        List<LeafReaderContext> leafReaderContexts = Arrays.asList(leafReaderContext1, leafReaderContext2);

        DocIdSetIterator docIdSetIterator = mock(DocIdSetIterator.class);
        when(docIdSetIterator.docID()).thenReturn(0, 1, DocIdSetIterator.NO_MORE_DOCS);
        Scorer scorer = mock(Scorer.class);
        when(scorer.iterator()).thenReturn(docIdSetIterator);
        when(scorer.docID()).thenReturn(0, 1, DocIdSetIterator.NO_MORE_DOCS);
        when(scorer.score()).thenReturn(10.f, 11.f, -1f);

        Weight weight = mock(Weight.class);
        when(weight.scorer(leafReaderContext1)).thenReturn(null);
        when(weight.scorer(leafReaderContext2)).thenReturn(scorer);

        ContextualProfileBreakdown profile = mock(ContextualProfileBreakdown.class);
        Timer timer = mock(Timer.class);
        when(profile.context(any())).thenReturn(profile);
        when(profile.getTimer(KNNQueryTimingType.ANN_SEARCH)).thenReturn(timer);

        // Run
        List<Map<Integer, Float>> results = queryUtils.doSearch(indexSearcher, leafReaderContexts, weight, profile);

        // Verify
        verify(profile, times(2)).context(any());
        verify(profile, times(2)).getTimer(KNNQueryTimingType.ANN_SEARCH);
        verify(timer, times(2)).start();
        verify(timer, times(2)).stop();

        assertEquals(2, results.size());
        assertEquals(0, results.get(0).size());
        assertEquals(2, results.get(1).size());
        assertEquals(10.f, results.get(1).get(0));
        assertEquals(11.f, results.get(1).get(1));

    }

    @SneakyThrows
    public void testGetAllSiblings_whenEmptyDocIds_thenEmptyIterator() {
        LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
        BitSetProducer bitSetProducer = mock(BitSetProducer.class);
        Bits bits = mock(Bits.class);

        // Run
        DocIdSetIterator docIdSetIterator = queryUtils.getAllSiblings(leafReaderContext, Collections.emptySet(), bitSetProducer, bits);

        // Verify
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, docIdSetIterator.nextDoc());
    }

    @SneakyThrows
    public void testGetAllSiblings_whenNonEmptyDocIds_thenReturnAllSiblings() {
        LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
        // 0, 1, 2(parent), 3, 4, 5, 6, 7(parent), 8, 9, 10(parent)
        BitSet bitSet = new FixedBitSet(new long[1], 11);
        bitSet.set(2);
        bitSet.set(7);
        bitSet.set(10);
        BitSetProducer bitSetProducer = mock(BitSetProducer.class);
        when(bitSetProducer.getBitSet(leafReaderContext)).thenReturn(bitSet);

        BitSet filterBits = new FixedBitSet(new long[1], 11);
        filterBits.set(1);
        filterBits.set(8);
        filterBits.set(9);

        // Run
        Set<Integer> docIds = Set.of(1, 8);
        DocIdSetIterator docIdSetIterator = queryUtils.getAllSiblings(leafReaderContext, docIds, bitSetProducer, filterBits);

        // Verify
        Set<Integer> expectedDocIds = Set.of(1, 8, 9);
        Set<Integer> returnedDocIds = new HashSet<>();
        docIdSetIterator.nextDoc();
        while (docIdSetIterator.docID() != DocIdSetIterator.NO_MORE_DOCS) {
            returnedDocIds.add(docIdSetIterator.docID());
            docIdSetIterator.nextDoc();
        }
        assertEquals(expectedDocIds, returnedDocIds);
    }

    @SneakyThrows
    public void testCreateBits_whenWeightIsNull_thenMatchAllBits() {
        LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);

        // Run
        Bits bits = queryUtils.createBits(leafReaderContext, null);

        // Verify
        assertEquals(Bits.MatchAllBits.class, bits.getClass());

    }

    @SneakyThrows
    public void testCreateBits_whenScoreIsNull_thenMatchNoBits() {
        LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
        Weight weight = mock(Weight.class);
        when(weight.scorer(leafReaderContext)).thenReturn(null);

        // Run
        Bits bits = queryUtils.createBits(leafReaderContext, weight);

        // Verify
        assertEquals(Bits.MatchNoBits.class, bits.getClass());
    }

    @SneakyThrows
    public void testCreateBits_whenCalled_thenReturnBits() {
        FixedBitSet liveDocBitSet = new FixedBitSet(new long[1], 11);
        liveDocBitSet.set(2);
        liveDocBitSet.set(7);
        liveDocBitSet.set(10);

        FixedBitSet matchedBitSet = new FixedBitSet(new long[1], 11);
        matchedBitSet.set(1);
        matchedBitSet.set(2);
        matchedBitSet.set(4);
        matchedBitSet.set(9);
        matchedBitSet.set(10);

        BitSetIterator matchedBitSetIterator = new BitSetIterator(matchedBitSet, 5);

        LeafReader leafReader = mock(LeafReader.class);
        when(leafReader.getLiveDocs()).thenReturn(liveDocBitSet);
        when(leafReader.maxDoc()).thenReturn(11);

        LeafReaderContext leafReaderContext = mock(LeafReaderContext.class);
        when(leafReaderContext.reader()).thenReturn(leafReader);

        Scorer scorer = mock(Scorer.class);
        when(scorer.iterator()).thenReturn(matchedBitSetIterator);

        Weight weight = mock(Weight.class);
        when(weight.scorer(leafReaderContext)).thenReturn(scorer);

        // Run
        Bits bits = queryUtils.createBits(leafReaderContext, weight);

        // Verify
        FixedBitSet expectedBitSet = matchedBitSet.clone();
        expectedBitSet.and(liveDocBitSet);
        assertTrue(areSetBitsEqual(expectedBitSet, bits));
    }

    private boolean areSetBitsEqual(Bits bits1, Bits bits2) {
        int minLength = Math.min(bits1.length(), bits2.length());

        for (int i = 0; i < minLength; i++) {
            if (bits1.get(i) != bits2.get(i)) {
                return false;
            }
        }

        for (int i = minLength; i < bits1.length(); i++) {
            if (bits1.get(i)) {
                return false;
            }
        }

        for (int i = minLength; i < bits2.length(); i++) {
            if (bits2.get(i)) {
                return false;
            }
        }

        return true;
    }
}

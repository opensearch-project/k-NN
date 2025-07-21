/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profile;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.util.IOSupplier;
import org.mockito.Mock;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.profile.ContextualProfileBreakdown;
import org.opensearch.search.profile.Timer;
import org.opensearch.search.profile.query.QueryProfiler;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.verify;
import static org.mockito.MockitoAnnotations.openMocks;

public class KNNProfileUtilTests extends OpenSearchTestCase {

    @Mock
    private IOSupplier<String> mockAction;
    @Mock
    private LeafReaderContext mockLeafContext;
    @Mock
    private Enum<?> mockTimingType;
    @Mock
    private ContextIndexSearcher mockContextSearcher;
    @Mock
    private QueryProfiler mockProfiler;
    @Mock
    private Query mockQuery;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        openMocks(this);
        when(mockAction.get()).thenReturn("test result");
    }

    public void testGetProfilerWithContextIndexSearcher() {
        when(mockContextSearcher.getProfiler()).thenReturn(mockProfiler);
        QueryProfiler result = KNNProfileUtil.getProfiler(mockContextSearcher);
        assertSame(mockProfiler, result);
    }

    public void testGetProfilerWithContextIndexSearcherNullProfiler() {
        when(mockContextSearcher.getProfiler()).thenReturn(null);
        QueryProfiler result = KNNProfileUtil.getProfiler(mockContextSearcher);
        assertNull(result);
    }

    public void testGetProfilerWithRegularIndexSearcher() {
        IndexSearcher mockSearcher = mock(IndexSearcher.class);
        QueryProfiler result = KNNProfileUtil.getProfiler(mockSearcher);
        assertNull(result);
    }

    public void testProfileWithQueryProfilerNonNull() throws IOException {
        ContextualProfileBreakdown mockBreakdown = mock(ContextualProfileBreakdown.class);
        Timer mockTimer = mock(Timer.class);

        when(mockProfiler.getProfileBreakdown(mockQuery)).thenReturn(mockBreakdown);
        when(mockBreakdown.context(mockLeafContext)).thenReturn(mockBreakdown);
        when(mockBreakdown.getTimer(mockTimingType)).thenReturn(mockTimer);

        Object result = KNNProfileUtil.profile(mockProfiler, mockQuery, mockLeafContext, mockTimingType, mockAction);
        assertEquals("test result", result);
    }

    public void testProfileWithQueryProfilerNull() throws IOException {
        Object result = KNNProfileUtil.profile(null, mockQuery, mockLeafContext, mockTimingType, mockAction);
        verify(mockAction).get();
        assertEquals("test result", result);
    }

    public void testProfileWithContextualProfileBreakdownNonNull() throws IOException {
        ContextualProfileBreakdown mockProfile = mock(ContextualProfileBreakdown.class);
        Timer mockTimer = mock(Timer.class);

        when(mockProfile.context(mockLeafContext)).thenReturn(mockProfile);
        when(mockProfile.getTimer(mockTimingType)).thenReturn(mockTimer);

        Object result = KNNProfileUtil.profile(mockProfile, mockLeafContext, mockTimingType, mockAction);
        verify(mockProfile).context(mockLeafContext);
        verify(mockProfile).getTimer(mockTimingType);
        verify(mockTimer).start();
        verify(mockTimer).stop();
        verify(mockAction).get();
        assertEquals("test result", result);
    }

    public void testProfileWithContextualProfileBreakdownNull() throws IOException {
        Object result = KNNProfileUtil.profile(null, mockLeafContext, mockTimingType, mockAction);
        verify(mockAction).get();
        assertEquals("test result", result);
    }
}

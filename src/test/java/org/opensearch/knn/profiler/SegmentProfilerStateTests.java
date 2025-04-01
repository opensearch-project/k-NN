/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profiler;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.lucene.search.DocIdSetIterator;
import org.junit.Before;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class SegmentProfilerStateTests extends OpenSearchTestCase {

    private KNNVectorValues<Object> mockVectorValues;
    private Supplier<KNNVectorValues<?>> mockSupplier;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        mockVectorValues = (KNNVectorValues<Object>) mock(KNNVectorValues.class);
        mockSupplier = () -> mockVectorValues;
    }

    public void testConstructor() {
        List<SummaryStatistics> statistics = new ArrayList<>();
        statistics.add(new SummaryStatistics());

        SegmentProfilerState state = new SegmentProfilerState(statistics);
        assertEquals(statistics, state.getStatistics());
    }

    public void testProfileVectorsWithNullVectorValues() throws IOException {
        Supplier<KNNVectorValues<?>> nullSupplier = () -> null;
        SegmentProfilerState state = SegmentProfilerState.profileVectors(nullSupplier);

        assertTrue(state.getStatistics().isEmpty());
    }

    public void testProfileVectorsWithNoDocuments() throws IOException {
        when(mockVectorValues.docId()).thenReturn(DocIdSetIterator.NO_MORE_DOCS);

        SegmentProfilerState state = SegmentProfilerState.profileVectors(mockSupplier);
        assertTrue(state.getStatistics().isEmpty());
    }

    public void testProfileVectorsWithSingleFloatVector() throws IOException {
        float[] vector = new float[] { 1.0f, 2.0f, 3.0f };

        when(mockVectorValues.docId()).thenReturn(0);
        when(mockVectorValues.dimension()).thenReturn(3);
        when(mockVectorValues.getVector()).thenReturn(vector);
        when(mockVectorValues.nextDoc()).thenReturn(DocIdSetIterator.NO_MORE_DOCS);

        SegmentProfilerState state = SegmentProfilerState.profileVectors(mockSupplier);

        assertEquals(3, state.getStatistics().size());
        assertEquals(1.0, state.getStatistics().get(0).getMean(), 0.001);
        assertEquals(2.0, state.getStatistics().get(1).getMean(), 0.001);
        assertEquals(3.0, state.getStatistics().get(2).getMean(), 0.001);
    }

    public void testProfileVectorsWithSingleByteVector() throws IOException {
        byte[] vector = new byte[] { 1, 2, 3 };

        when(mockVectorValues.docId()).thenReturn(0);
        when(mockVectorValues.dimension()).thenReturn(3);
        when(mockVectorValues.getVector()).thenReturn(vector);
        when(mockVectorValues.nextDoc()).thenReturn(DocIdSetIterator.NO_MORE_DOCS);

        SegmentProfilerState state = SegmentProfilerState.profileVectors(mockSupplier);

        assertEquals(3, state.getStatistics().size());
        assertEquals(1.0, state.getStatistics().get(0).getMean(), 0.001);
        assertEquals(2.0, state.getStatistics().get(1).getMean(), 0.001);
        assertEquals(3.0, state.getStatistics().get(2).getMean(), 0.001);
    }

    public void testProfileVectorsWithMultipleFloatVectors() throws IOException {
        float[] vector1 = new float[] { 1.0f, 2.0f };
        float[] vector2 = new float[] { 3.0f, 4.0f };

        when(mockVectorValues.docId()).thenReturn(0);
        when(mockVectorValues.dimension()).thenReturn(2);
        when(mockVectorValues.getVector()).thenReturn(vector1).thenReturn(vector2);
        when(mockVectorValues.nextDoc()).thenReturn(1).thenReturn(DocIdSetIterator.NO_MORE_DOCS);

        SegmentProfilerState state = SegmentProfilerState.profileVectors(mockSupplier);

        assertEquals(2, state.getStatistics().size());
        assertEquals(2.0, state.getStatistics().get(0).getMean(), 0.001);
        assertEquals(3.0, state.getStatistics().get(1).getMean(), 0.001);
    }

    public void testProfileVectorsWithMultipleByteVectors() throws IOException {
        byte[] vector1 = new byte[] { 1, 2 };
        byte[] vector2 = new byte[] { 3, 4 };

        when(mockVectorValues.docId()).thenReturn(0);
        when(mockVectorValues.dimension()).thenReturn(2);
        when(mockVectorValues.getVector()).thenReturn(vector1).thenReturn(vector2);
        when(mockVectorValues.nextDoc()).thenReturn(1).thenReturn(DocIdSetIterator.NO_MORE_DOCS);

        SegmentProfilerState state = SegmentProfilerState.profileVectors(mockSupplier);

        assertEquals(2, state.getStatistics().size());
        assertEquals(2.0, state.getStatistics().get(0).getMean(), 0.001);
        assertEquals(3.0, state.getStatistics().get(1).getMean(), 0.001);
    }

    public void testProfileVectorsStatisticalValues() throws IOException {
        float[] vector1 = new float[] { 1.0f, 2.0f };
        float[] vector2 = new float[] { 3.0f, 4.0f };
        float[] vector3 = new float[] { 5.0f, 6.0f };

        when(mockVectorValues.docId()).thenReturn(0);
        when(mockVectorValues.dimension()).thenReturn(2);
        when(mockVectorValues.getVector()).thenReturn(vector1).thenReturn(vector2).thenReturn(vector3);
        when(mockVectorValues.nextDoc()).thenReturn(1).thenReturn(2).thenReturn(DocIdSetIterator.NO_MORE_DOCS);

        SegmentProfilerState state = SegmentProfilerState.profileVectors(mockSupplier);

        assertEquals(3.0, state.getStatistics().get(0).getMean(), 0.001);
        assertEquals(2.0, state.getStatistics().get(0).getStandardDeviation(), 0.001);
        assertEquals(1.0, state.getStatistics().get(0).getMin(), 0.001);
        assertEquals(5.0, state.getStatistics().get(0).getMax(), 0.001);

        assertEquals(4.0, state.getStatistics().get(1).getMean(), 0.001);
        assertEquals(2.0, state.getStatistics().get(1).getStandardDeviation(), 0.001);
        assertEquals(2.0, state.getStatistics().get(1).getMin(), 0.001);
        assertEquals(6.0, state.getStatistics().get(1).getMax(), 0.001);
    }

    public void testProfileVectorsWithByteStatisticalValues() throws IOException {
        byte[] vector1 = new byte[] { 1, 2 };
        byte[] vector2 = new byte[] { 3, 4 };
        byte[] vector3 = new byte[] { 5, 6 };

        when(mockVectorValues.docId()).thenReturn(0);
        when(mockVectorValues.dimension()).thenReturn(2);
        when(mockVectorValues.getVector()).thenReturn(vector1).thenReturn(vector2).thenReturn(vector3);
        when(mockVectorValues.nextDoc()).thenReturn(1).thenReturn(2).thenReturn(DocIdSetIterator.NO_MORE_DOCS);

        SegmentProfilerState state = SegmentProfilerState.profileVectors(mockSupplier);

        assertEquals(3.0, state.getStatistics().get(0).getMean(), 0.001);
        assertEquals(2.0, state.getStatistics().get(0).getStandardDeviation(), 0.001);
        assertEquals(1.0, state.getStatistics().get(0).getMin(), 0.001);
        assertEquals(5.0, state.getStatistics().get(0).getMax(), 0.001);

        assertEquals(4.0, state.getStatistics().get(1).getMean(), 0.001);
        assertEquals(2.0, state.getStatistics().get(1).getStandardDeviation(), 0.001);
        assertEquals(2.0, state.getStatistics().get(1).getMin(), 0.001);
        assertEquals(6.0, state.getStatistics().get(1).getMax(), 0.001);
    }
}

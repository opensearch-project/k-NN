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
    private static final String TEST_SEGMENT_ID = "test_segment";

    @Before
    public void setUp() throws Exception {
        super.setUp();
        mockVectorValues = (KNNVectorValues<Object>) mock(KNNVectorValues.class);
        mockSupplier = () -> mockVectorValues;
    }

    public void testConstructor() {
        List<SummaryStatistics> statistics = new ArrayList<>();
        statistics.add(new SummaryStatistics());
        int dimension = 1;

        SegmentProfilerState state = new SegmentProfilerState(statistics, dimension, TEST_SEGMENT_ID);
        assertEquals(statistics, state.getStatistics());
        assertEquals(dimension, state.getDimension());
        assertEquals(TEST_SEGMENT_ID, state.getSegmentId());
    }

    public void testProfileVectorsWithNullVectorValues() throws IOException {
        Supplier<KNNVectorValues<?>> nullSupplier = () -> null;
        SegmentProfilerState state = SegmentProfilerState.profileVectors(nullSupplier, TEST_SEGMENT_ID);

        assertTrue(state.getStatistics().isEmpty());
        assertEquals(0, state.getDimension());
        assertEquals(TEST_SEGMENT_ID, state.getSegmentId());
    }

    public void testProfileVectorsWithNoDocuments() throws IOException {
        when(mockVectorValues.docId()).thenReturn(DocIdSetIterator.NO_MORE_DOCS);
        when(mockVectorValues.dimension()).thenReturn(3);

        SegmentProfilerState state = SegmentProfilerState.profileVectors(mockSupplier, TEST_SEGMENT_ID);
        assertTrue(state.getStatistics().isEmpty());
        assertEquals(3, state.getDimension());
        assertEquals(TEST_SEGMENT_ID, state.getSegmentId());
    }

    public void testProfileVectorsWithSingleFloatVector() throws IOException {
        float[] vector = new float[] { 1.0f, 2.0f, 3.0f };
        int dimension = 3;

        when(mockVectorValues.docId()).thenReturn(0);
        when(mockVectorValues.dimension()).thenReturn(dimension);
        when(mockVectorValues.getVector()).thenReturn(vector);
        when(mockVectorValues.nextDoc()).thenReturn(DocIdSetIterator.NO_MORE_DOCS);

        SegmentProfilerState state = SegmentProfilerState.profileVectors(mockSupplier, TEST_SEGMENT_ID);

        assertEquals(dimension, state.getDimension());
        assertEquals(dimension, state.getStatistics().size());
        assertEquals(TEST_SEGMENT_ID, state.getSegmentId());
        assertEquals(1.0, state.getStatistics().get(0).getMean(), 0.001);
        assertEquals(2.0, state.getStatistics().get(1).getMean(), 0.001);
        assertEquals(3.0, state.getStatistics().get(2).getMean(), 0.001);
    }

    public void testProfileVectorsWithSingleByteVector() throws IOException {
        byte[] vector = new byte[] { 1, 2, 3 };
        int dimension = 3;

        when(mockVectorValues.docId()).thenReturn(0);
        when(mockVectorValues.dimension()).thenReturn(dimension);
        when(mockVectorValues.getVector()).thenReturn(vector);
        when(mockVectorValues.nextDoc()).thenReturn(DocIdSetIterator.NO_MORE_DOCS);

        SegmentProfilerState state = SegmentProfilerState.profileVectors(mockSupplier, TEST_SEGMENT_ID);

        assertEquals(dimension, state.getDimension());
        assertEquals(dimension, state.getStatistics().size());
        assertEquals(TEST_SEGMENT_ID, state.getSegmentId());
        assertEquals(1.0, state.getStatistics().get(0).getMean(), 0.001);
        assertEquals(2.0, state.getStatistics().get(1).getMean(), 0.001);
        assertEquals(3.0, state.getStatistics().get(2).getMean(), 0.001);
    }

    public void testSerializationDeserialization() {
        List<SummaryStatistics> statistics = new ArrayList<>();
        SummaryStatistics stats = new SummaryStatistics();
        stats.addValue(1.0);
        stats.addValue(2.0);
        statistics.add(stats);

        SegmentProfilerState originalState = new SegmentProfilerState(statistics, 1, TEST_SEGMENT_ID);
        byte[] serialized = originalState.toByteArray();
        SegmentProfilerState deserializedState = SegmentProfilerState.fromBytes(serialized);

        assertEquals(originalState.getDimension(), deserializedState.getDimension());
        assertEquals(originalState.getSegmentId(), deserializedState.getSegmentId());
        assertEquals(originalState.getStatistics().size(), deserializedState.getStatistics().size());
        assertEquals(originalState.getStatistics().get(0).getMean(), deserializedState.getStatistics().get(0).getMean(), 0.001);
    }

    public void testProfileVectorsStatisticalValues() throws IOException {
        float[] vector1 = new float[] { 1.0f, 2.0f };
        float[] vector2 = new float[] { 3.0f, 4.0f };
        float[] vector3 = new float[] { 5.0f, 6.0f };
        int dimension = 2;

        when(mockVectorValues.docId()).thenReturn(0);
        when(mockVectorValues.dimension()).thenReturn(dimension);
        when(mockVectorValues.getVector()).thenReturn(vector1).thenReturn(vector2).thenReturn(vector3);
        when(mockVectorValues.nextDoc()).thenReturn(1).thenReturn(2).thenReturn(DocIdSetIterator.NO_MORE_DOCS);

        SegmentProfilerState state = SegmentProfilerState.profileVectors(mockSupplier, TEST_SEGMENT_ID);

        assertEquals(dimension, state.getDimension());
        assertEquals(TEST_SEGMENT_ID, state.getSegmentId());

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

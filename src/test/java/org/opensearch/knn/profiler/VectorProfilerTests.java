/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profiler;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.mockito.MockitoAnnotations;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

public class VectorProfilerTests extends OpenSearchTestCase {

    private AutoCloseable mocks;
    private VectorProfiler profiler;
    private List<float[]> testVectors;
    private static final String TEST_FIELD = "test_field";

    @Before
    public void setUp() throws Exception {
        super.setUp();
        mocks = MockitoAnnotations.openMocks(this);
        profiler = VectorProfiler.getInstance();
        profiler.setFieldToDimensionStats(new HashMap<>());
        testVectors = Arrays.asList(new float[] { 1.0f, 2.0f, 3.0f }, new float[] { 4.0f, 5.0f, 6.0f }, new float[] { 7.0f, 8.0f, 9.0f });
    }

    @After
    public void tearDown() throws Exception {
        super.tearDown();
        mocks.close();
    }

    @Test
    public void testSingletonBehavior() {
        VectorProfiler instance1 = VectorProfiler.getInstance();
        VectorProfiler instance2 = VectorProfiler.getInstance();
        assertSame(instance1, instance2);
    }

    @Test
    public void testProcessVectors() {
        profiler.processVectors(TEST_FIELD, testVectors);
        List<DimensionStatisticAggregator> stats = profiler.getFieldStatistics(TEST_FIELD);

        assertNotNull(stats);
        assertEquals(3, stats.size());

        assertEquals(4.0, stats.get(0).getAggregateStatistics().getMean(), 0.001);
        assertEquals(5.0, stats.get(1).getAggregateStatistics().getMean(), 0.001);
        assertEquals(6.0, stats.get(2).getAggregateStatistics().getMean(), 0.001);
    }

    @Test
    public void testMultipleFields() {
        String field1 = "field1";
        String field2 = "field2";

        profiler.processVectors(field1, testVectors);
        profiler.processVectors(field2, testVectors);

        assertNotSame(profiler.getFieldStatistics(field1), profiler.getFieldStatistics(field2));
    }

    @Test
    public void testEmptyVectors() {
        Exception ex = assertThrows(IllegalArgumentException.class, () -> profiler.processVectors(TEST_FIELD, Collections.emptyList()));
        assertEquals("Vectors collection cannot be null or empty", ex.getMessage());
    }

    @Test
    public void testDifferentDimensions() {
        List<float[]> vectorsWith2Dimensions = Arrays.asList(new float[] { 1.0f, 2.0f }, new float[] { 3.0f, 4.0f });

        profiler.processVectors(TEST_FIELD, vectorsWith2Dimensions);
        List<DimensionStatisticAggregator> stats = profiler.getFieldStatistics(TEST_FIELD);

        assertEquals("Number of dimension aggregators should match vector dimension", 2, stats.size());
        assertEquals(2.0, stats.get(0).getAggregateStatistics().getMean(), 0.001);
        assertEquals(3.0, stats.get(1).getAggregateStatistics().getMean(), 0.001);
    }

    @Test
    public void testLargeVectors() {
        List<float[]> largeVectors = new ArrayList<>();
        int numVectors = 1000;
        for (int i = 0; i < numVectors; i++) {
            largeVectors.add(new float[] { i, i, i });
        }

        profiler.processVectors(TEST_FIELD, largeVectors);
        List<DimensionStatisticAggregator> stats = profiler.getFieldStatistics(TEST_FIELD);

        double expectedMean = (numVectors - 1) / 2.0;
        assertEquals(expectedMean, stats.get(0).getAggregateStatistics().getMean(), 3.0);
    }

    @Test
    public void testNullVectors() {
        Exception ex = assertThrows(IllegalArgumentException.class, () -> profiler.processVectors(TEST_FIELD, null));
        assertEquals("Vectors collection cannot be null or empty", ex.getMessage());
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profiler;

import org.apache.commons.math3.stat.descriptive.StatisticalSummary;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.mockito.MockitoAnnotations;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class VectorProfilerTests extends OpenSearchTestCase {

    private AutoCloseable mocks;
    private VectorProfiler profiler;
    private List<float[]> testVectors;
    private List<Computation> aggregateComputations;
    private static final String TEST_FIELD = "test_field";

    @Before
    public void setUp() throws Exception {
        super.setUp();
        mocks = MockitoAnnotations.openMocks(this);
        profiler = VectorProfiler.getInstance();

        testVectors = Arrays.asList(new float[] { 1.0f, 2.0f, 3.0f }, new float[] { 4.0f, 5.0f, 6.0f }, new float[] { 7.0f, 8.0f, 9.0f });

        aggregateComputations = profiler.getRegisteredComputations();
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

        VectorProfiler mockInstance = new VectorProfiler();
        VectorProfiler.setInstance(mockInstance);
        assertSame(mockInstance, VectorProfiler.getInstance());
    }

    @Test
    public void testFieldBasedComputation() {
        Map<Computation, float[]> results = profiler.sampleAndCompute(TEST_FIELD, testVectors);

        float[] mean = results.get(aggregateComputations.get(0));
        assertArrayEquals(new float[] { 4.0f, 5.0f, 6.0f }, mean, 0.001f);
    }

    @Test
    public void testMultipleFieldsComputation() {
        String field1 = "field1";
        String field2 = "field2";

        profiler.sampleAndCompute(field1, testVectors);
        profiler.sampleAndCompute(field2, testVectors);

        Map<Computation, float[]> results1 = profiler.sampleAndCompute(field1, testVectors);
        Map<Computation, float[]> results2 = profiler.sampleAndCompute(field2, testVectors);

        assertNotSame(results1, results2);
    }

    @Test
    public void testCustomComputation() {
        Computation maxComputation = StatisticalSummary::getMax;
        profiler.registerComputation(maxComputation);

        Map<Computation, float[]> results = profiler.sampleAndCompute(TEST_FIELD, testVectors);

        float[] maxValues = results.get(maxComputation);
        assertArrayEquals(new float[] { 7.0f, 8.0f, 9.0f }, maxValues, 0.001f);
    }

    @Test
    public void testEmptyVectors() {
        Exception ex = assertThrows(IllegalArgumentException.class, () -> profiler.sampleAndCompute(TEST_FIELD, Collections.emptyList()));
        assertEquals("Vectors collection cannot be null or empty", ex.getMessage());
    }

    @Test
    public void testDifferentDimensions() {
        List<float[]> differentDimVectors = Arrays.asList(new float[] { 1.0f, 2.0f }, new float[] { 3.0f });

        Map<Computation, float[]> results = profiler.sampleAndCompute(TEST_FIELD, differentDimVectors);
        assertEquals(2, results.get(aggregateComputations.get(0)).length);
    }

    @Test
    public void testLargeVectors() {
        List<float[]> largeVectors = new ArrayList<>();
        int numVectors = 1000;
        for (int i = 0; i < numVectors; i++) {
            largeVectors.add(new float[] { i, i, i });
        }

        Map<Computation, float[]> results = profiler.sampleAndCompute(TEST_FIELD, largeVectors);

        float[] mean = results.get(aggregateComputations.get(0));
        double expectedMean = numVectors / 2.0;
        double tolerance = expectedMean * 0.20;

        assertTrue(Math.abs(mean[0] - expectedMean) <= tolerance);
    }

    @Test
    public void testComputationManagement() {
        int initialSize = profiler.getRegisteredComputations().size();
        Computation newComp = StatisticalSummary::getMin;

        profiler.registerComputation(newComp);
        assertEquals(initialSize + 1, profiler.getRegisteredComputations().size());

        profiler.unregisterComputation(newComp);
        assertEquals(initialSize, profiler.getRegisteredComputations().size());
    }
}

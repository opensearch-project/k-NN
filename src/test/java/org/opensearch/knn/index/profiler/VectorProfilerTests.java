/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.profiler;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.opensearch.knn.profiler.Computation;
import org.opensearch.knn.profiler.StatisticalOperators;
import org.opensearch.knn.profiler.VectorProfiler;
import org.opensearch.test.OpenSearchTestCase;
import org.opensearch.threadpool.ThreadPool;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class VectorProfilerTests extends OpenSearchTestCase {

    @Mock
    private ThreadPool threadPool;

    private AutoCloseable mocks;
    private VectorProfiler profiler;
    private List<float[]> testVectors;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        mocks = MockitoAnnotations.openMocks(this);
        profiler = VectorProfiler.getInstance();
        profiler.initialize();

        // Create test vectors
        testVectors = Arrays.asList(
                new float[]{1.0f, 2.0f, 3.0f},
                new float[]{4.0f, 5.0f, 6.0f},
                new float[]{7.0f, 8.0f, 9.0f}
        );
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
    public void testCustomComputation() {
        Computation maxComputation = perDimensionVector -> {
            if (perDimensionVector == null || perDimensionVector.length == 0) {
                return new float[]{0.0f};
            }
            float max = perDimensionVector[0];
            for (float value : perDimensionVector) {
                max = Math.max(max, value);
            }
            return new float[]{max};
        };

        profiler.registerComputation(maxComputation);
        Map<Computation, float[]> results = profiler.sampleAndCompute(testVectors);

        float[] maxValues = results.get(maxComputation);
        assertNotNull(maxValues);
        assertArrayEquals(new float[]{7.0f, 8.0f, 9.0f}, maxValues, 0.001f);
    }

    @Test
    public void testDefaultComputations() {
        Map<Computation, float[]> results = profiler.sampleAndCompute(testVectors);

        // Test mean calculation
        float[] mean = results.get(StatisticalOperators.MEAN);
        assertArrayEquals(new float[]{4.0f, 5.0f, 6.0f}, mean, 0.001f);

        // Test variance calculation
        float[] variance = results.get(StatisticalOperators.VARIANCE);
        assertArrayEquals(new float[]{9.0f, 9.0f, 9.0f}, variance, 0.001f);

        // Test standard deviation calculation
        float[] stdDev = results.get(StatisticalOperators.STANDARD_DEVIATION);
        assertArrayEquals(new float[]{3.0f, 3.0f, 3.0f}, stdDev, 0.001f);
    }

    @Test
    public void testSampleAndCompute_EmptyVectors() {
        Exception ex = assertThrows(
                IllegalArgumentException.class,
                () -> profiler.sampleAndCompute(Collections.emptyList())
        );
        assertEquals("Vectors collection cannot be null or empty", ex.getMessage());
    }

    @Test
    public void testSampleAndCompute_DifferentDimensions() {
        List<float[]> differentDimVectors = Arrays.asList(
                new float[]{1.0f, 2.0f},
                new float[]{3.0f}
        );

        Map<Computation, float[]> results = profiler.sampleAndCompute(differentDimVectors);
        assertNotNull(results.get(StatisticalOperators.MEAN));
        assertEquals(2, results.get(StatisticalOperators.MEAN).length);
    }

    @Test
    public void testSampleAndCompute_CustomSampleSize() {
        int sampleSize = 2;
        Map<Computation, float[]> results = profiler.sampleAndCompute(testVectors, sampleSize);

        assertNotNull(results.get(StatisticalOperators.MEAN));
        assertNotNull(results.get(StatisticalOperators.VARIANCE));
        assertNotNull(results.get(StatisticalOperators.STANDARD_DEVIATION));
    }

    @Test
    public void testSampleAndCompute_NullVectors() {
        Exception ex = assertThrows(
                IllegalArgumentException.class,
                () -> profiler.sampleAndCompute(null)
        );
        assertEquals("Vectors collection cannot be null or empty", ex.getMessage());
    }

    @Test
    public void testSingleVector() {
        List<float[]> singleVector = Collections.singletonList(new float[]{1.0f, 2.0f, 3.0f});
        Map<Computation, float[]> results = profiler.sampleAndCompute(singleVector);

        assertArrayEquals(singleVector.get(0), results.get(StatisticalOperators.MEAN), 0.001f);
        assertArrayEquals(new float[]{0.0f, 0.0f, 0.0f}, results.get(StatisticalOperators.VARIANCE), 0.001f);
        assertArrayEquals(new float[]{0.0f, 0.0f, 0.0f}, results.get(StatisticalOperators.STANDARD_DEVIATION), 0.001f);
    }

    @Test
    public void testLargeVectors() {
        List<float[]> largeVectors = new ArrayList<>();
        for (int i = 0; i < 1000; i++) {
            largeVectors.add(new float[]{i, i, i});
        }

        Map<Computation, float[]> results = profiler.sampleAndCompute(largeVectors);
        assertNotNull(results.get(StatisticalOperators.MEAN));
        assertNotNull(results.get(StatisticalOperators.VARIANCE));
        assertNotNull(results.get(StatisticalOperators.STANDARD_DEVIATION));
    }

    @Test
    public void testComputationManagement() {
        int initialSize = profiler.getRegisteredComputations().size();

        Computation newComp = perDimensionVector -> new float[]{perDimensionVector[0]};

        profiler.registerComputation(newComp);
        assertEquals(initialSize + 1, profiler.getRegisteredComputations().size());

        profiler.unregisterComputation(newComp);
        assertEquals(initialSize, profiler.getRegisteredComputations().size());
    }
}
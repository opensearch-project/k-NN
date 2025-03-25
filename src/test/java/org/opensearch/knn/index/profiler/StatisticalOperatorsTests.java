/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.profiler;

import org.junit.Test;
import org.opensearch.knn.profiler.StatisticalOperators;

import static org.junit.Assert.assertEquals;

/**
 * Test class for StatisticalOperators.
 * This class contains unit tests to verify the functionality of the StatisticalOperators class,
 * which implements the Computation interface for statistical calculations on floating point values.
 */
public class StatisticalOperatorsTests {

    private final StatisticalOperators mean = StatisticalOperators.MEAN;
    private final StatisticalOperators variance = StatisticalOperators.VARIANCE;
    private final StatisticalOperators stdDev = StatisticalOperators.STANDARD_DEVIATION;

    @Test
    public void testMeanComputation() {
        float[] values = new float[] { 2.0f, 4.0f, 6.0f };
        float[] result = mean.compute(values);
        assertEquals(4.0f, result[0], 0.0001);
    }

    @Test
    public void testMeanWithSingleValue() {
        float[] values = new float[] { 5.0f };
        float[] result = mean.compute(values);
        assertEquals(5.0f, result[0], 0.0001);
    }

    @Test
    public void testMeanWithZeroValues() {
        float[] values = new float[] { 0.0f, 0.0f, 0.0f };
        float[] result = mean.compute(values);
        assertEquals(0.0f, result[0], 0.0001);
    }

    @Test
    public void testVarianceComputation() {
        float[] values = new float[] { 2.0f, 4.0f, 6.0f };
        float[] result = variance.compute(values);
        assertEquals(4.0f, result[0], 0.0001);
    }

    @Test
    public void testVarianceWithSingleValue() {
        float[] values = new float[] { 5.0f };
        float[] result = variance.compute(values);
        assertEquals(0.0f, result[0], 0.0001);
    }

    @Test
    public void testVarianceWithIdenticalValues() {
        float[] values = new float[] { 2.0f, 2.0f, 2.0f };
        float[] result = variance.compute(values);
        assertEquals(0.0f, result[0], 0.0001);
    }

    @Test
    public void testStandardDeviationComputation() {
        float[] values = new float[] { 2.0f, 4.0f, 6.0f };
        float[] result = stdDev.compute(values);
        assertEquals(2.0f, result[0], 0.0001);
    }

    @Test
    public void testStandardDeviationWithSingleValue() {
        float[] values = new float[] { 5.0f };
        float[] result = stdDev.compute(values);
        assertEquals(0.0f, result[0], 0.0001);
    }

    @Test
    public void testStandardDeviationWithIdenticalValues() {
        float[] values = new float[] { 2.0f, 2.0f, 2.0f };
        float[] result = stdDev.compute(values);
        assertEquals(0.0f, result[0], 0.0001);
    }

    @Test
    public void testWithNegativeValues() {
        float[] values = new float[] { -2.0f, 0.0f, 2.0f };
        float[] meanResult = mean.compute(values);
        float[] varianceResult = variance.compute(values);
        float[] stdDevResult = stdDev.compute(values);

        assertEquals(0.0f, meanResult[0], 0.0001);
        assertEquals(4.0f, varianceResult[0], 0.0001);
        assertEquals(2.0f, stdDevResult[0], 0.0001);
    }

    @Test
    public void testWithEmptyArray() {
        float[] values = new float[] {};
        float[] meanResult = mean.compute(values);
        float[] varianceResult = variance.compute(values);
        float[] stdDevResult = stdDev.compute(values);

        assertEquals(0.0f, meanResult[0], 0.0001);
        assertEquals(0.0f, varianceResult[0], 0.0001);
        assertEquals(0.0f, stdDevResult[0], 0.0001);
    }

    @Test
    public void testWithNullInput() {
        float[] meanResult = mean.compute(null);
        float[] varianceResult = variance.compute(null);
        float[] stdDevResult = stdDev.compute(null);

        assertEquals(0.0f, meanResult[0], 0.0001);
        assertEquals(0.0f, varianceResult[0], 0.0001);
        assertEquals(0.0f, stdDevResult[0], 0.0001);
    }

    @Test
    public void testLargeNumbers() {
        float[] values = new float[] { 1000.0f, 2000.0f, 3000.0f };
        float[] meanResult = mean.compute(values);
        float[] varianceResult = variance.compute(values);
        float[] stdDevResult = stdDev.compute(values);

        assertEquals(2000.0f, meanResult[0], 0.0001);
        assertEquals(1000000.0f, varianceResult[0], 0.0001);
        assertEquals(1000.0f, stdDevResult[0], 0.0001);
    }
}

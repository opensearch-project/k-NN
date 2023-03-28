/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.memory.breaker;

import org.opensearch.common.unit.ByteSizeUnit;
import org.opensearch.common.unit.ByteSizeValue;
import org.opensearch.knn.KNNTestCase;

import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNSettings.KNN_CIRCUIT_BREAKER_TRIGGERED;
import static org.opensearch.knn.index.KNNSettings.KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE;
import static org.opensearch.knn.index.KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_ENABLED;
import static org.opensearch.knn.index.KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_LIMIT;

public class NativeMemoryCircuitBreakerTests extends KNNTestCase {

    public void testIsTripped() {
        boolean isTripped = randomBoolean();
        when(knnSettings.getSettingValue(KNN_CIRCUIT_BREAKER_TRIGGERED)).thenReturn(isTripped);
        NativeMemoryCircuitBreaker nativeMemoryCircuitBreaker = new NativeMemoryCircuitBreaker(knnSettings);
        assertEquals(isTripped, nativeMemoryCircuitBreaker.isTripped());
    }

    public void testSet() {
        boolean isTripped = randomBoolean();
        doNothing().when(knnSettings).updateBooleanSetting(KNN_CIRCUIT_BREAKER_TRIGGERED, isTripped);
        NativeMemoryCircuitBreaker nativeMemoryCircuitBreaker = new NativeMemoryCircuitBreaker(knnSettings);
        nativeMemoryCircuitBreaker.set(isTripped);
        verify(knnSettings, times(1)).updateBooleanSetting(KNN_CIRCUIT_BREAKER_TRIGGERED, isTripped);
    }

    public void testGetLimit() {
        ByteSizeValue circuitBreakerLimit = new ByteSizeValue(randomIntBetween(10, 10000), ByteSizeUnit.KB);
        when(knnSettings.getSettingValue(KNN_MEMORY_CIRCUIT_BREAKER_LIMIT)).thenReturn(circuitBreakerLimit);
        NativeMemoryCircuitBreaker nativeMemoryCircuitBreaker = new NativeMemoryCircuitBreaker(knnSettings);
        assertEquals(circuitBreakerLimit, nativeMemoryCircuitBreaker.getLimit());
    }

    public void testIsEnabled() {
        boolean isEnabled = randomBoolean();
        when(knnSettings.getSettingValue(KNN_MEMORY_CIRCUIT_BREAKER_ENABLED)).thenReturn(isEnabled);
        NativeMemoryCircuitBreaker nativeMemoryCircuitBreaker = new NativeMemoryCircuitBreaker(knnSettings);
        assertEquals(isEnabled, nativeMemoryCircuitBreaker.isEnabled());
    }

    public void testGetUnsetPercentage() {
        double unsetPercentage = 71;
        when(knnSettings.getSettingValue(KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE)).thenReturn(unsetPercentage);
        NativeMemoryCircuitBreaker nativeMemoryCircuitBreaker = new NativeMemoryCircuitBreaker(knnSettings);
        assertEquals(unsetPercentage, nativeMemoryCircuitBreaker.getUnsetPercentage(), 0.0001);
    }
}

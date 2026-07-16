/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profile.query;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.profile.LongMetric;
import org.opensearch.search.profile.ProfileMetric;
import org.opensearch.search.profile.Timer;

import java.util.Collection;
import java.util.Locale;
import java.util.function.Supplier;

public class KNNMetricsTests extends KNNTestCase {

    public void testGetKNNQueryMetrics_containsAllTimingTypesAndCardinality() {
        Collection<Supplier<ProfileMetric>> metrics = KNNMetrics.getKNNQueryMetrics();

        assertEquals(KNNQueryTimingType.values().length + 1, metrics.size());

        long timerCount = metrics.stream().filter(s -> s.get() instanceof Timer).count();
        long longMetricCount = metrics.stream().filter(s -> s.get() instanceof LongMetric).count();

        assertEquals(KNNQueryTimingType.values().length, timerCount);
        assertEquals(1, longMetricCount);
    }

    public void testGetKNNQueryMetrics_createsTimerForEachTimingType() {
        Collection<Supplier<ProfileMetric>> metrics = KNNMetrics.getKNNQueryMetrics();

        for (KNNQueryTimingType timingType : KNNQueryTimingType.values()) {
            boolean found = metrics.stream().anyMatch(s -> {
                ProfileMetric metric = s.get();
                return metric instanceof Timer && timingType.toString().equals(metric.getName());
            });
            assertTrue("Missing timer for " + timingType, found);
        }
    }

    public void testGetKNNQueryMetrics_createsCardinalityMetric() {
        Collection<Supplier<ProfileMetric>> metrics = KNNMetrics.getKNNQueryMetrics();

        boolean found = metrics.stream().anyMatch(s -> {
            ProfileMetric metric = s.get();
            return metric instanceof LongMetric && KNNMetrics.CARDINALITY.equals(metric.getName());
        });
        assertTrue(found);
    }

    public void testGetNativeMetrics_includesNestedDocsMetric() {
        Collection<Supplier<ProfileMetric>> nativeMetrics = KNNMetrics.getNativeMetrics();
        Collection<Supplier<ProfileMetric>> queryMetrics = KNNMetrics.getKNNQueryMetrics();

        assertEquals(queryMetrics.size() + 1, nativeMetrics.size());

        boolean found = nativeMetrics.stream().anyMatch(s -> {
            ProfileMetric metric = s.get();
            return metric instanceof LongMetric && KNNMetrics.NUM_NESTED_DOCS.equals(metric.getName());
        });
        assertTrue(found);
    }

    public void testKNNQueryTimingTypeToString() {
        assertEquals("ann_search", KNNQueryTimingType.ANN_SEARCH.toString());
        assertEquals("exact_search", KNNQueryTimingType.EXACT_SEARCH.toString());
        assertEquals("graph_load", KNNQueryTimingType.GRAPH_LOAD.toString());
        assertEquals("bitset_creation", KNNQueryTimingType.BITSET_CREATION.toString());
    }

    public void testKNNQueryTimingTypeValues() {
        assertEquals(4, KNNQueryTimingType.values().length);
        assertEquals(KNNQueryTimingType.ANN_SEARCH, KNNQueryTimingType.valueOf("ANN_SEARCH"));
    }

    public void testKNNQueryTimingTypeToStringUsesRootLocale() {
        for (KNNQueryTimingType type : KNNQueryTimingType.values()) {
            assertEquals(type.name().toLowerCase(Locale.ROOT), type.toString());
        }
    }
}

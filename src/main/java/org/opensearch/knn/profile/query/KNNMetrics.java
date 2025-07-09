/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profile.query;

import lombok.Getter;
import lombok.Setter;
import org.opensearch.knn.profile.LongMetric;
import org.opensearch.search.profile.ProfileMetric;
import org.opensearch.search.profile.Profilers;
import org.opensearch.search.profile.Timer;

import java.util.ArrayList;
import java.util.Collection;
import java.util.function.Supplier;

public class KNNMetrics {

    public static final String NUM_NESTED_DOCS = "num_nested_docs";
    public static final String CARDINALITY = "cardinality";

    @Getter
    @Setter
    private static Profilers profilers;

    public static Collection<Supplier<ProfileMetric>> getKNNQueryMetrics() {
        Collection<Supplier<ProfileMetric>> metrics = new ArrayList<>();
        for (KNNQueryTimingType type : KNNQueryTimingType.values()) {
            metrics.add(() -> new Timer(type.toString()));
        }

        metrics.add(() -> new LongMetric(CARDINALITY));

        return metrics;
    }

    public static Collection<Supplier<ProfileMetric>> getNativeMetrics() {
        Collection<Supplier<ProfileMetric>> metrics = getKNNQueryMetrics();
        for (NativeEngineKnnTimingType type : NativeEngineKnnTimingType.values()) {
            metrics.add(() -> new Timer(type.toString()));
        }

        metrics.add(() -> new LongMetric(NUM_NESTED_DOCS));

        return metrics;
    }

    public static Collection<Supplier<ProfileMetric>> getLuceneMetrics() {
        Collection<Supplier<ProfileMetric>> metrics = new ArrayList<>();
        for (LuceneEngineKnnTimingType type : LuceneEngineKnnTimingType.values()) {
            metrics.add(() -> new Timer(type.toString()));
        }

        return metrics;
    }
}

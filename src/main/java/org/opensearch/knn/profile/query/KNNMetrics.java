/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profile.query;

import org.opensearch.knn.index.query.nativelib.NativeEngineKnnVectorQuery;
import org.opensearch.knn.profile.LongMetric;
import org.opensearch.search.profile.ProfileMetric;
import org.opensearch.search.profile.Timer;

import java.util.ArrayList;
import java.util.Collection;
import java.util.function.Supplier;

/**
 *  Container for {@link org.opensearch.search.profile.ProfileMetric}s based on query type
 */
public class KNNMetrics {

    public static final String NUM_NESTED_DOCS = "num_nested_docs";
    public static final String CARDINALITY = "cardinality";

    /**
     * Contains profile metric information for KNN Queries based on {@link KNNQueryTimingType} timers. Additionally, it
     * contains a metric for filter cardinality.
     * @return list of {@link org.opensearch.search.profile.ProfileMetric} for KNNQueries
     *
     */
    public static Collection<Supplier<ProfileMetric>> getKNNQueryMetrics() {
        Collection<Supplier<ProfileMetric>> metrics = new ArrayList<>();
        for (KNNQueryTimingType type : KNNQueryTimingType.values()) {
            metrics.add(() -> new Timer(type.toString()));
        }

        metrics.add(() -> new LongMetric(CARDINALITY));

        return metrics;
    }

    /**
     * Contains profile metric information for KNN queries. Additionally, contains a metric on the number
     * of nested docs for {@link NativeEngineKnnVectorQuery} queries.
     * @return list of {@link org.opensearch.search.profile.ProfileMetric} for NativeEngineQueries
     */
    public static Collection<Supplier<ProfileMetric>> getNativeMetrics() {
        Collection<Supplier<ProfileMetric>> metrics = getKNNQueryMetrics();

        metrics.add(() -> new LongMetric(NUM_NESTED_DOCS));

        return metrics;
    }
}

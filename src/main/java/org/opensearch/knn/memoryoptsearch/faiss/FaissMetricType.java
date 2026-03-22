/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.opensearch.knn.index.SpaceType;

/**
 * Maps Faiss C++ {@code MetricType} enum values to their Java equivalents.
 *
 * <p>Ordinals match the Faiss C++ definition: {@code METRIC_INNER_PRODUCT = 0}, {@code METRIC_L2 = 1}.
 * This is used when reading the metric type from a serialized Faiss binary index header
 * (see {@code FaissBinaryIndex#readBinaryCommonHeader(IndexInput)}) and converting it to the corresponding
 * {@link SpaceType} for search-time scoring.
 * See MetricType.h for more details.
 *
 * @see FaissFlatIndexFactory#maybeSetFlatBinaryIndex
 */
public enum FaissMetricType {
    /** Faiss {@code METRIC_INNER_PRODUCT} (ordinal 0). */
    INNER_PRODUCT(SpaceType.INNER_PRODUCT),
    /** Faiss {@code METRIC_L2} (ordinal 1). */
    L2(SpaceType.L2);

    FaissMetricType(final SpaceType spaceType) {
        this.spaceType = spaceType;
    }

    public final SpaceType spaceType;
}

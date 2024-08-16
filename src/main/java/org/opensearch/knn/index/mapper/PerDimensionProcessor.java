/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

/**
 * Process values per dimension. Good to have if we want to do some kind of cleanup on data as it is coming in.
 */
public interface PerDimensionProcessor {

    /**
     * Process float value per dimension.
     *
     * @param value value to process
     * @return processed value
     */
    default float process(float value) {
        return value;
    }

    /**
     * Process byte as float value per dimension.
     *
     * @param value value to process
     * @return processed value
     */
    default float processByte(float value) {
        return value;
    }

    PerDimensionProcessor NOOP_PROCESSOR = new PerDimensionProcessor() {
    };
}

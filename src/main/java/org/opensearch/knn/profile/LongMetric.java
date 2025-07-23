/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profile;

import lombok.Getter;
import lombok.Setter;
import org.opensearch.search.profile.ProfileMetric;

import java.util.Map;

/**
 * Sets and returns a long value for profiling.
 */
public class LongMetric extends ProfileMetric {

    @Getter
    @Setter
    private Long value;

    public LongMetric(String name) {
        super(name);
        value = 0L;
    }

    @Override
    public Map<String, Long> toBreakdownMap() {
        return Map.of(getName(), value);
    }
}

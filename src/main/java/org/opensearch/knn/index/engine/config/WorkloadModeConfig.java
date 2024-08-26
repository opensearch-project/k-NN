/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.config;

import lombok.AllArgsConstructor;
import lombok.Getter;

import static org.opensearch.knn.common.KNNConstants.MODE_IN_MEMORY_NAME;
import static org.opensearch.knn.common.KNNConstants.MODE_ON_DISK_NAME;

@AllArgsConstructor
@Getter
public enum WorkloadModeConfig {
    NOT_CONFIGURED("NA"),
    IN_MEMORY(MODE_IN_MEMORY_NAME),
    ON_DISK(MODE_ON_DISK_NAME);

    public static final WorkloadModeConfig DEFAULT = IN_MEMORY;

    public static WorkloadModeConfig fromString(String name) {
        if (name == null || name.equals("NA")) {
            return NOT_CONFIGURED;
        }

        if (name.equalsIgnoreCase(IN_MEMORY.name)) {
            return IN_MEMORY;
        }

        if (name.equalsIgnoreCase(ON_DISK.name)) {
            return ON_DISK;
        }
        throw new IllegalArgumentException("Invalid workload mode: " + name);
    }

    private final String name;

    @Override
    public String toString() {
        return name;
    }
}

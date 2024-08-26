/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.config;

import lombok.AllArgsConstructor;
import lombok.Getter;

@AllArgsConstructor
@Getter
public enum CompressionConfig {
    NOT_CONFIGURED(-1),
    x1(1),
    x2(2),
    x4(4),
    x8(8),
    x16(16),
    x32(32);

    public static final CompressionConfig DEFAULT = x1;

    public static CompressionConfig fromString(String name) {
        if (name == null || name.equals("NA")) {
            return NOT_CONFIGURED;
        }

        for (CompressionConfig config : CompressionConfig.values()) {
            if (config.toString().equals(name)) {
                return config;
            }
        }
        throw new IllegalArgumentException("Invalid compression level: " + name);
    }

    private final int compressionLevel;

    @Override
    public String toString() {
        if (this == NOT_CONFIGURED) {
            return "NA";
        }
        return "x" + compressionLevel;
    }
}

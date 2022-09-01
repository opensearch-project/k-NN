/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import lombok.Getter;
import lombok.Setter;

/**
 * Class defines collection of flags for stats
 */

public enum KNNFlag {
    BUILT_WITH_FAISS,
    BUILT_WITH_LUCENE,
    BUILT_WITH_NMSLIB;

    @Getter
    @Setter
    private boolean value = false;
}

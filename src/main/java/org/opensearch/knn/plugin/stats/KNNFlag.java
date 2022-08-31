/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Class defines collection of flags for stats
 */
public enum KNNFlag {
    BUILT_WITH_FAISS,
    BUILT_WITH_LUCENE,
    BUILT_WITH_NMSLIB;

    private final AtomicBoolean flag = new AtomicBoolean(false);

    /**
     * Set the value of a flag
     *
     * @param value counter value
     */
    public void set(boolean value) {
        flag.getAndSet(value);
    }

    /**
     * Get the value of flag
     *
     * @return flag
     */
    public boolean getFlag() {
        return flag.get();
    }
}

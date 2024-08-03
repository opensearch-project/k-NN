/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.util;

import lombok.experimental.UtilityClass;

/**
 * Utility class to manage version information in a thread-safe manner using ThreadLocal storage.
 * This class ensures that version information is available within the current thread context.
 */
@UtilityClass
public class VersionContext {

    /**
     * ThreadLocal storage for version information.
     * This allows each thread to have its own version information without interference.
     */
    private final ThreadLocal<Integer> versionHolder = new ThreadLocal<>();

    /**
     * Sets the version for the current thread.
     *
     * @param version the version to be set.
     */
    public void setVersion(int version) {
        versionHolder.set(version);
    }

    /**
     * Gets the version for the current thread.
     *
     * @return the version for the current thread.
     */
    public int getVersion() {
        return versionHolder.get();
    }

    /**
     * Clears the version for the current thread.
     */
    public void clear() {
        versionHolder.remove();
    }
}

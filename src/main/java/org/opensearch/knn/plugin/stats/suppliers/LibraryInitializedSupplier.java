/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.plugin.stats.suppliers;

import org.opensearch.knn.index.engine.KNNLibrary;

import java.util.function.Supplier;

/**
 * Supplier to determine whether library has been initialized
 */
public class LibraryInitializedSupplier implements Supplier<Boolean> {
    private KNNLibrary knnLibrary;

    /**
     * Constructor
     *
     * @param knnLibrary to check if initialized
     */
    public LibraryInitializedSupplier(KNNLibrary knnLibrary) {
        this.knnLibrary = knnLibrary;
    }

    @Override
    public Boolean get() {
        return knnLibrary.isInitialized();
    }
}

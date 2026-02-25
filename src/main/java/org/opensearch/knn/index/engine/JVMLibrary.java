/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import java.util.Map;

/**
 * Abstract class for JVM based KNN libraries
 */
public abstract class JVMLibrary extends AbstractKNNLibrary {

    boolean initialized;

    /**
     * Constructor
     *
     * @param methods Map of k-NN methods that the library supports
     * @param version String representing version of library
     */
    public JVMLibrary(Map<String, KNNMethod> methods, String version) {
        super(methods, version);
    }

    @Override
    public int estimateOverheadInKB(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
        throw new UnsupportedOperationException("Estimating overhead is not supported for JVM based libraries.");
    }

    @Override
    public Boolean isInitialized() {
        return initialized;
    }

    @Override
    public void setInitialized(Boolean isInitialized) {
        initialized = isInitialized;
    }
}

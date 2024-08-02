/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.Builder;

import java.util.Map;

/**
 * Simple implementation of {@link KNNLibraryIndexBuildContext}
 */
@Builder
public class KNNLibraryIndexBuildContextImpl implements KNNLibraryIndexBuildContext {

    private Map<String, Object> parameters;

    @Override
    public Map<String, Object> getLibraryParameters() {
        return parameters;
    }
}

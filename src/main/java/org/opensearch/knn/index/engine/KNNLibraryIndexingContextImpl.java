/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.Builder;
import org.opensearch.knn.index.mapper.PerDimensionProcessor;
import org.opensearch.knn.index.mapper.PerDimensionValidator;
import org.opensearch.knn.index.mapper.VectorValidator;

import java.util.Map;

/**
 * Simple implementation of {@link KNNLibraryIndexingContext}
 */
@Builder
public class KNNLibraryIndexingContextImpl implements KNNLibraryIndexingContext {

    private VectorValidator vectorValidator;
    private PerDimensionValidator perDimensionValidator;
    private PerDimensionProcessor perDimensionProcessor;
    private Map<String, Object> parameters;

    @Override
    public Map<String, Object> getLibraryParameters() {
        return parameters;
    }

    @Override
    public VectorValidator getVectorValidator() {
        return vectorValidator;
    }

    @Override
    public PerDimensionValidator getPerDimensionValidator() {
        return perDimensionValidator;
    }

    @Override
    public PerDimensionProcessor getPerDimensionProcessor() {
        return perDimensionProcessor;
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.Builder;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.mapper.PerDimensionProcessor;
import org.opensearch.knn.index.mapper.PerDimensionValidator;
import org.opensearch.knn.index.mapper.VectorValidator;

import java.util.Collections;
import java.util.Map;
import java.util.function.Function;

/**
 * Simple implementation of {@link KNNLibraryIndexingContext}
 */
@Builder
public class KNNLibraryIndexingContextImpl implements KNNLibraryIndexingContext {

    private VectorValidator vectorValidator;
    private PerDimensionValidator perDimensionValidator;
    private PerDimensionProcessor perDimensionProcessor;
    @Builder.Default
    private Map<String, Object> parameters = Collections.emptyMap();
    @Builder.Default
    private QuantizationConfig quantizationConfig = QuantizationConfig.EMPTY;
    private Function<TrainingConfigValidationInput, TrainingConfigValidationOutput> trainingConfigValidationSetup;

    @Override
    public Map<String, Object> getLibraryParameters() {
        return parameters;
    }

    @Override
    public QuantizationConfig getQuantizationConfig() {
        return quantizationConfig;
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

    @Override
    public Function<TrainingConfigValidationInput, TrainingConfigValidationOutput> getTrainingConfigValidationSetup() {
        return trainingConfigValidationSetup;
    }
}

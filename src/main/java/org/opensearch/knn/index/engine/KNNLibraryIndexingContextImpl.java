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
import java.util.function.BiFunction;

import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;

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
    public BiFunction<Long, KNNMethodContext, TrainingConfigValidationOutput> getTrainingConfigValidationSetup() {
        return (trainingVectors, knnMethodContext) -> {

            long minTrainingVectorCount = 1000;
            TrainingConfigValidationOutput.TrainingConfigValidationOutputBuilder builder = TrainingConfigValidationOutput.builder();

            MethodComponentContext encoderContext = (MethodComponentContext) knnMethodContext.getMethodComponentContext()
                .getParameters()
                .get(METHOD_ENCODER_PARAMETER);

            if (knnMethodContext.getMethodComponentContext().getParameters().containsKey(METHOD_PARAMETER_NLIST)
                && encoderContext.getParameters().containsKey(ENCODER_PARAMETER_PQ_CODE_SIZE)) {

                int nlist = ((Integer) knnMethodContext.getMethodComponentContext().getParameters().get(METHOD_PARAMETER_NLIST));
                int code_size = ((Integer) encoderContext.getParameters().get(ENCODER_PARAMETER_PQ_CODE_SIZE));
                minTrainingVectorCount = (long) Math.max(nlist, Math.pow(2, code_size));
            }

            if (trainingVectors < minTrainingVectorCount) {
                builder.valid(false).minTrainingVectorCount(minTrainingVectorCount);
                return builder.build();
            }

            builder.valid(true);
            return builder.build();
        };
    }
}

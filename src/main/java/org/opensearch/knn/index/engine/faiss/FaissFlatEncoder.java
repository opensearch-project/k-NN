/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import com.google.common.collect.ImmutableSet;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_M;
import org.opensearch.knn.index.engine.TrainingConfigValidationInput;
import org.opensearch.knn.index.engine.TrainingConfigValidationOutput;
import org.opensearch.knn.index.engine.KNNMethodContext;

/**
 * Flat faiss encoder. Flat encoding means that it does nothing. It needs an encoder, though, because it
 * is used in generating the index description.
 */
public class FaissFlatEncoder implements Encoder {

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(
        VectorDataType.FLOAT,
        VectorDataType.BYTE,
        VectorDataType.BINARY
    );

    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(KNNConstants.ENCODER_FLAT)
        .setKnnLibraryIndexingContextGenerator(
            ((methodComponent, methodComponentContext, knnMethodConfigContext) -> MethodAsMapBuilder.builder(
                KNNConstants.FAISS_FLAT_DESCRIPTION,
                methodComponent,
                methodComponentContext,
                knnMethodConfigContext
            ).build())
        )
        .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
        .build();

    @Override
    public MethodComponent getMethodComponent() {
        return METHOD_COMPONENT;
    }

    @Override
    public CompressionLevel calculateCompressionLevel(
        MethodComponentContext encoderContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        return CompressionLevel.x1;
    }

    @Override
    public TrainingConfigValidationOutput validateEncoderConfig(TrainingConfigValidationInput trainingConfigValidationInput) {
        KNNMethodContext knnMethodContext = trainingConfigValidationInput.getKnnMethodContext();
        KNNMethodConfigContext knnMethodConfigContext = trainingConfigValidationInput.getKnnMethodConfigContext();

        TrainingConfigValidationOutput.TrainingConfigValidationOutputBuilder builder = TrainingConfigValidationOutput.builder();

        // validate ENCODER_PARAMETER_PQ_M is divisible by vector dimension
        if (knnMethodContext != null && knnMethodConfigContext != null) {
            if (knnMethodContext.getMethodComponentContext().getParameters().containsKey(ENCODER_PARAMETER_PQ_M)
                && knnMethodConfigContext.getDimension() % (Integer) knnMethodContext.getMethodComponentContext()
                    .getParameters()
                    .get(ENCODER_PARAMETER_PQ_M) != 0) {
                builder.valid(false);
                return builder.build();
            } else {
                builder.valid(true);
            }
        }
        return builder.build();
    }
}

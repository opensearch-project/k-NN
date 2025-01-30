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

import org.opensearch.knn.index.engine.TrainingConfigValidationInput;
import org.opensearch.knn.index.engine.TrainingConfigValidationOutput;

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
        TrainingConfigValidationOutput.TrainingConfigValidationOutputBuilder builder = TrainingConfigValidationOutput.builder();
        return builder.build();
    }
}

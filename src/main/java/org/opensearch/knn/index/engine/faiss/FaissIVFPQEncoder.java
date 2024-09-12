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
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.BYTES_PER_KILOBYTES;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_COUNT_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_COUNT_LIMIT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE_LIMIT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_M;
import static org.opensearch.knn.common.KNNConstants.FAISS_PQ_DESCRIPTION;

/**
 * Faiss IVF PQ encoder. Right now, the implementations are slightly different during validation between this an
 * {@link FaissHNSWPQEncoder}. Hence, they are separate classes.
 */
public class FaissIVFPQEncoder implements Encoder {

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT);

    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(KNNConstants.ENCODER_PQ)
        .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
        .addParameter(
            ENCODER_PARAMETER_PQ_M,
            new Parameter.IntegerParameter(ENCODER_PARAMETER_PQ_M, ENCODER_PARAMETER_PQ_CODE_COUNT_DEFAULT, (v, context) -> {
                boolean isValueGreaterThan0 = v > 0;
                boolean isValueLessThanCodeCountLimit = v < ENCODER_PARAMETER_PQ_CODE_COUNT_LIMIT;
                boolean isDimensionDivisibleByValue = context.getDimension() % v == 0;
                return isValueGreaterThan0 && isValueLessThanCodeCountLimit && isDimensionDivisibleByValue;
            })
        )
        .addParameter(
            ENCODER_PARAMETER_PQ_CODE_SIZE,
            new Parameter.IntegerParameter(ENCODER_PARAMETER_PQ_CODE_SIZE, ENCODER_PARAMETER_PQ_CODE_SIZE_DEFAULT, (v, context) -> {
                boolean isValueGreaterThan0 = v > 0;
                boolean isValueLessThanCodeSizeLimit = v < ENCODER_PARAMETER_PQ_CODE_SIZE_LIMIT;
                return isValueGreaterThan0 && isValueLessThanCodeSizeLimit;
            })
        )
        .setRequiresTraining(true)
        .setKnnLibraryIndexingContextGenerator(
            ((methodComponent, methodComponentContext, knnMethodConfigContext) -> MethodAsMapBuilder.builder(
                FAISS_PQ_DESCRIPTION,
                methodComponent,
                methodComponentContext,
                knnMethodConfigContext
            ).addParameter(ENCODER_PARAMETER_PQ_M, "", "").addParameter(ENCODER_PARAMETER_PQ_CODE_SIZE, "x", "").build())
        )
        .setOverheadInKBEstimator((methodComponent, methodComponentContext, dimension) -> {
            // Size estimate formula: (4 * d * 2^code_size) / 1024 + 1

            // Get value of code size passed in by user
            Object codeSizeObject = methodComponentContext.getParameters().get(ENCODER_PARAMETER_PQ_CODE_SIZE);

            // If not specified, get default value of code size
            if (codeSizeObject == null) {
                Parameter<?> codeSizeParameter = methodComponent.getParameters().get(ENCODER_PARAMETER_PQ_CODE_SIZE);
                if (codeSizeParameter == null) {
                    throw new IllegalStateException(
                        String.format("%s  is not a valid parameter. This is a bug.", ENCODER_PARAMETER_PQ_CODE_SIZE)
                    );
                }

                codeSizeObject = codeSizeParameter.getDefaultValue();
            }

            if (!(codeSizeObject instanceof Integer)) {
                throw new IllegalStateException(String.format("%s must be an integer.", ENCODER_PARAMETER_PQ_CODE_SIZE));
            }

            int codeSize = (Integer) codeSizeObject;
            return ((4L * (1L << codeSize) * dimension) / BYTES_PER_KILOBYTES) + 1;
        })
        .build();

    @Override
    public MethodComponent getMethodComponent() {
        return METHOD_COMPONENT;
    }

    @Override
    public CompressionLevel calculateCompressionLevel(
        MethodComponentContext methodComponentContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        // TODO: For now, not supported out of the box
        return CompressionLevel.NOT_CONFIGURED;
    }
}

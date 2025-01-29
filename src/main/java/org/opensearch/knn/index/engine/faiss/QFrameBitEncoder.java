/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import com.google.common.collect.ImmutableSet;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContextImpl;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.index.engine.TrainingConfigValidationInput;
import org.opensearch.knn.index.engine.TrainingConfigValidationOutput;
import org.opensearch.knn.index.engine.KNNMethodContext;

import java.util.HashMap;
import java.util.Locale;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.FAISS_FLAT_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_M;

/**
 * Quantization framework binary encoder,
 */
public class QFrameBitEncoder implements Encoder {

    public static final String NAME = "binary";
    public static final String BITCOUNT_PARAM = "bits";
    private static final int DEFAULT_BITS = 1;
    private static final Set<Integer> validBitCounts = ImmutableSet.of(1, 2, 4);
    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT);

    /**
     * {
     *   "encoder": {
     *     "name": "binary",
     *     "parameters": {
     *       "bits": 2
     *     }
     *   }
     * }
     */
    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(NAME)
        .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
        .addParameter(
            BITCOUNT_PARAM,
            new Parameter.IntegerParameter(BITCOUNT_PARAM, DEFAULT_BITS, (v, context) -> validBitCounts.contains(v))
        )
        .setKnnLibraryIndexingContextGenerator(((methodComponent, methodComponentContext, knnMethodConfigContext) -> {
            QuantizationConfig quantizationConfig;
            int bitCount = (int) methodComponentContext.getParameters().getOrDefault(BITCOUNT_PARAM, DEFAULT_BITS);
            if (bitCount == 1) {
                quantizationConfig = QuantizationConfig.builder().quantizationType(ScalarQuantizationType.ONE_BIT).build();
            } else if (bitCount == 2) {
                quantizationConfig = QuantizationConfig.builder().quantizationType(ScalarQuantizationType.TWO_BIT).build();
            } else if (bitCount == 4) {
                quantizationConfig = QuantizationConfig.builder().quantizationType(ScalarQuantizationType.FOUR_BIT).build();
            } else {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "Invalid bit count: %d", bitCount));
            }

            // We use the flat description because we are doing the quantization
            return KNNLibraryIndexingContextImpl.builder().quantizationConfig(quantizationConfig).parameters(new HashMap<>() {
                {
                    put(INDEX_DESCRIPTION_PARAMETER, FAISS_FLAT_DESCRIPTION);
                }
            }).build();
        }))
        .setRequiresTraining(false)
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
        if (methodComponentContext.getParameters().containsKey(BITCOUNT_PARAM) == false) {
            return CompressionLevel.NOT_CONFIGURED;
        }

        // Map the number of bits passed in, back to the compression level
        Object value = methodComponentContext.getParameters().get(BITCOUNT_PARAM);
        ValidationException validationException = METHOD_COMPONENT.getParameters()
            .get(BITCOUNT_PARAM)
            .validate(value, knnMethodConfigContext);
        if (validationException != null) {
            throw validationException;
        }

        Integer bitCount = (Integer) value;
        if (bitCount == 1) {
            return CompressionLevel.x32;
        }

        if (bitCount == 2) {
            return CompressionLevel.x16;
        }

        // Validation will ensure that only 1 of the supported bit count will be selected.
        return CompressionLevel.x8;
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

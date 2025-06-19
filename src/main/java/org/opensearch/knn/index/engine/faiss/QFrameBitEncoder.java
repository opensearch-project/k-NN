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

import java.util.HashMap;
import java.util.Locale;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.FAISS_FLAT_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;

/**
 * Quantization framework binary encoder,
 */
public class QFrameBitEncoder implements Encoder {

    public static final String NAME = "binary";
    public static final String BITCOUNT_PARAM = "bits";
    private static final int DEFAULT_BITS = 1;
    public static final String ENABLE_ADC_PARAM = "enable_adc";
    public static final Boolean DEFAULT_ENABLE_ADC = false;
    public static final String ENABLE_RANDOM_ROTATION_PARAM = "random_rotation";
    public static final Boolean DEFAULT_ENABLE_RANDOM_ROTATION = false;
    private static final Set<Integer> validBitCounts = ImmutableSet.of(1, 2, 4);
    private static final Set<Integer> supportedBitCountsForADC = ImmutableSet.of(1);
    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT);

    /**
     * {
     *   "encoder": {
     *     "name": "binary",
     *     "parameters": {
     *       "bits": 2,
     *       "random_rotation": true,
     *       "enable_adc": false
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

        .addParameter(
            ENABLE_RANDOM_ROTATION_PARAM,
            new Parameter.BooleanParameter(ENABLE_RANDOM_ROTATION_PARAM, DEFAULT_ENABLE_RANDOM_ROTATION, (v, context) -> {
                return true; // all booleans are valid for this toggleable setting.
            })
        )
        .addParameter(ENABLE_ADC_PARAM, new Parameter.BooleanParameter(ENABLE_ADC_PARAM, DEFAULT_ENABLE_ADC, (v, context) -> {
            // all booleans are valid for this toggleable setting. However, ADC is only supported for certain bit counts.
            // That validation is handled as part of the knnLibraryIndexingContextGenerator builder logic below.
            return true;
        }))
        .setKnnLibraryIndexingContextGenerator(((methodComponent, methodComponentContext, knnMethodConfigContext) -> {

            QuantizationConfig.QuantizationConfigBuilder quantizationConfigBuilder = QuantizationConfig.builder();

            int bitCount = (int) methodComponentContext.getParameters().getOrDefault(BITCOUNT_PARAM, DEFAULT_BITS);
            boolean enableRandomRotation = (boolean) methodComponentContext.getParameters()
                .getOrDefault(ENABLE_RANDOM_ROTATION_PARAM, DEFAULT_ENABLE_RANDOM_ROTATION);

            boolean enableADC = (boolean) methodComponentContext.getParameters().getOrDefault(ENABLE_ADC_PARAM, DEFAULT_ENABLE_ADC);

            if (enableADC && !supportedBitCountsForADC.contains(bitCount)) {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "Validation Failed: ADC is not supported for bit count: %d", bitCount)
                );
            }

            ScalarQuantizationType quantizationType = switch (bitCount) {
                case 1 -> ScalarQuantizationType.ONE_BIT;
                case 2 -> ScalarQuantizationType.TWO_BIT;
                case 4 -> ScalarQuantizationType.FOUR_BIT;
                default -> throw new IllegalArgumentException(String.format(Locale.ROOT, "Invalid bit count: %d", bitCount));
            };

            QuantizationConfig quantizationConfig = quantizationConfigBuilder.quantizationType(quantizationType)
                    .enableRandomRotation(enableRandomRotation)
                    .enableADC(enableADC)
                .build();

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
}

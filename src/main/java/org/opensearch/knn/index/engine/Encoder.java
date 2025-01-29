/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.knn.index.mapper.CompressionLevel;

/**
 * Interface representing an encoder. An encoder generally refers to a vector quantizer.
 */
public interface Encoder {
    /**
     * The name of the encoder does not have to be unique. Howevwer, when using within a method, there cannot be
     * 2 encoders with the same name.
     *
     * @return Name of the encoder
     */
    default String getName() {
        return getMethodComponent().getName();
    }

    /**
     *
     * @return Method component associated with the encoder
     */
    MethodComponent getMethodComponent();

    /**
     * Calculate the compression level for the give params. Assume float32 vectors are used. All parameters should
     * be resolved in the encoderContext passed in.
     *
     * @param encoderContext Context for the encoder to extract params from
     * @return Compression level this encoder produces. If the encoder does not support this calculation yet, it will
     *          return {@link CompressionLevel#NOT_CONFIGURED}
     */
    CompressionLevel calculateCompressionLevel(MethodComponentContext encoderContext, KNNMethodConfigContext knnMethodConfigContext);

    /**
     * Validates config of encoder
     *
     * @return Validation output of encoder parameters
     */
    TrainingConfigValidationOutput validateEncoderConfig(TrainingConfigValidationInput validationInput);
}

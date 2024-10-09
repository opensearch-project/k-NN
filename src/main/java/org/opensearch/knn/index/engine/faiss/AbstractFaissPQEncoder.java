/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.CompressionLevel;

import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_M;

/**
 * Abstract class for Faiss PQ encoders. This class provides the common logic for product quantization based encoders
 */
public abstract class AbstractFaissPQEncoder implements Encoder {

    @Override
    public CompressionLevel calculateCompressionLevel(
        MethodComponentContext methodComponentContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        // Roughly speaking, PQ can be configured to produce a lot of different compression levels. The "m" parameter
        // specifies how many sub-vectors to break the vector up into, and then the "code_size" represents the number
        // of bits to encode each subvector. Thus, a d-dimensional vector of float32s goes from
        // d*32 -> (m)*code_size bits. So if we want (d*32)/(m*code_size) will be the compression level.
        //
        // Example:
        // d=768, m=384, code_size=8
        // (768*32)/(384*8) = 8x (i.e. 24,576 vs. 3,072).
        //
        // Because of this variability, we will need to properly round to one of the supported values.
        if (methodComponentContext.getParameters().containsKey(ENCODER_PARAMETER_PQ_M) == false
            || methodComponentContext.getParameters().containsKey(ENCODER_PARAMETER_PQ_CODE_SIZE) == false) {
            return CompressionLevel.NOT_CONFIGURED;
        }

        // Map the number of bits passed in, back to the compression level
        Object value = methodComponentContext.getParameters().get(ENCODER_PARAMETER_PQ_M);
        ValidationException validationException = getMethodComponent().getParameters()
            .get(ENCODER_PARAMETER_PQ_M)
            .validate(value, knnMethodConfigContext);
        if (validationException != null) {
            throw validationException;
        }
        Integer m = (Integer) value;
        value = methodComponentContext.getParameters().get(ENCODER_PARAMETER_PQ_CODE_SIZE);
        validationException = getMethodComponent().getParameters()
            .get(ENCODER_PARAMETER_PQ_CODE_SIZE)
            .validate(value, knnMethodConfigContext);
        if (validationException != null) {
            throw validationException;
        }
        Integer codeSize = (Integer) value;
        int dimension = knnMethodConfigContext.getDimension();

        float actualCompression = ((float) dimension * 32) / (m * codeSize);

        if (actualCompression < 2.0f) {
            return CompressionLevel.x1;
        }

        if (actualCompression < 4.0f) {
            return CompressionLevel.x2;
        }

        if (actualCompression < 8.0f) {
            return CompressionLevel.x4;
        }

        if (actualCompression < 16.0f) {
            return CompressionLevel.x8;
        }

        if (actualCompression < 32.0f) {
            return CompressionLevel.x16;
        }

        if (actualCompression < 64.0f) {
            return CompressionLevel.x32;
        }

        // TODO: The problem is that the theoretical compression level of PQ can be in the thousands. Thus, Im not sure
        // it makes sense to have an enum all the way up to that value. So, for now, we will just return the max
        // compression
        return CompressionLevel.MAX_COMPRESSION_LEVEL;
    }
}

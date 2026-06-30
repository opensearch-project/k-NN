/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Locale;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.ENCODER_BINARY;
import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PQ;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;

/**
 * Interface representing an encoder. An encoder generally refers to a vector quantizer.
 *
 * <p>Serves a dual role:
 * <ul>
 *   <li>Type classification (via {@link EncoderType} and {@link QuantizationBits}) that drives
 *       ResolvedIndexSpec behavioral decisions.</li>
 *   <li>Method component integration (via {@link #getMethodComponent()}) for the existing
 *       resolution framework.</li>
 * </ul>
 */
public interface Encoder {

    /**
     * Identifies the encoder type. Anchored to KNNConstants encoder name strings.
     */
    enum EncoderType {
        FLAT(ENCODER_FLAT),
        SQ(ENCODER_SQ),
        PQ(ENCODER_PQ),
        BQ(ENCODER_BINARY);

        private final String name;

        EncoderType(String name) {
            this.name = name;
        }

        public String getName() {
            return name;
        }

        public static EncoderType fromName(String name) {
            for (EncoderType type : values()) {
                if (type.name.equals(name)) {
                    return type;
                }
            }
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Unsupported encoder type: [%s]", name));
        }
    }

    /**
     * Unified quantization bits enum. Maps bit widths to compression levels.
     * Serialization-compatible: getValue() returns the int stored in FieldInfo/SQConfig.
     */
    enum QuantizationBits {
        ONE(1, CompressionLevel.x32),
        TWO(2, CompressionLevel.x16),
        FOUR(4, CompressionLevel.x8),
        SEVEN(7, CompressionLevel.x4),
        SIXTEEN(16, CompressionLevel.x2),
        /** Identity value for FLAT encoders: full precision float32 with no quantization applied. */
        FULL_PRECISION(32, CompressionLevel.x1),
        /** For PQ codebook quantization, where bits-per-dimension does not apply. */
        NOT_APPLICABLE(-1, CompressionLevel.NOT_CONFIGURED);

        private final int value;
        private final CompressionLevel compressionLevel;

        QuantizationBits(int value, CompressionLevel compressionLevel) {
            this.value = value;
            this.compressionLevel = compressionLevel;
        }

        public int getValue() {
            return value;
        }

        public CompressionLevel getCompressionLevel() {
            return compressionLevel;
        }

        public static QuantizationBits fromValue(int value) {
            for (QuantizationBits bits : values()) {
                if (bits.value == value) {
                    return bits;
                }
            }
            return FULL_PRECISION;
        }
    }

    /**
     * The name of the encoder does not have to be unique. However, when using within a method, there cannot be
     * 2 encoders with the same name.
     *
     * @return Name of the encoder
     */
    default String getName() {
        return getMethodComponent().getName();
    }

    /**
     * @return Method component associated with the encoder
     */
    MethodComponent getMethodComponent();

    /**
     * Calculate the compression level for the given params. Assume float32 vectors are used. All parameters should
     * be resolved in the encoderContext passed in.
     *
     * @param encoderContext Context for the encoder to extract params from
     * @param knnMethodConfigContext method config context
     * @return Compression level this encoder produces. If the encoder does not support this calculation yet, it will
     *          return {@link CompressionLevel#NOT_CONFIGURED}
     */
    CompressionLevel calculateCompressionLevel(MethodComponentContext encoderContext, KNNMethodConfigContext knnMethodConfigContext);

    /**
     * Validates config of encoder
     *
     * @param validationInput input for validation
     * @return Validation output of encoder parameters
     */
    default TrainingConfigValidationOutput validateEncoderConfig(TrainingConfigValidationInput validationInput) {
        TrainingConfigValidationOutput.TrainingConfigValidationOutputBuilder builder = TrainingConfigValidationOutput.builder();
        return builder.build();
    }

    /**
     * @return the encoder type classification
     */
    EncoderType getEncoderType();

    /**
     * @return the set of bit widths this encoder supports
     * TODO: Evaluate removal -- zero production callers, risks inconsistency with getQuantizationBits().
     */
    Set<QuantizationBits> getSupportedBits();

    /**
     * @return the resolved quantization bits for this encoder instance
     */
    default QuantizationBits getQuantizationBits() {
        return QuantizationBits.FULL_PRECISION;
    }
}

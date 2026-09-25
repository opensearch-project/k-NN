/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.VectorDataType;
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
        FULL_PRECISION(32, CompressionLevel.x1);

        /**
         * Integer-coded SQ widths, for either data type: FLOAT reaches them at x32/x16/x8 and
         * HALF_FLOAT at x16/x8/x4. SEVEN (Lucene 7-bit), SIXTEEN (Faiss fp16) and FULL_PRECISION
         * are not integer-coded and take their own paths.
         */
        private static final Set<QuantizationBits> SQ_CODED_BITS = Set.of(ONE, TWO, FOUR);

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

        /**
         * Reverse of {@link #getCompressionLevel()}: maps a {@link CompressionLevel} back to the bit
         * width used to achieve it (x32→1, x16→2, x8→4, x4→7, x2→16, x1→32). Falls back to
         * {@link #FULL_PRECISION} for {@link CompressionLevel#NOT_CONFIGURED} and any unmapped value.
         */
        public static QuantizationBits fromCompressionLevel(CompressionLevel compressionLevel) {
            for (QuantizationBits bits : values()) {
                if (bits.compressionLevel == compressionLevel) {
                    return bits;
                }
            }
            return FULL_PRECISION;
        }

        /**
         * True when {@code compressionLevel} quantizes {@code vectorDataType} to 1, 2 or 4 bits per
         * dimension: x32/x16/x8 for FLOAT, x16/x8/x4 for HALF_FLOAT. False for every other level,
         * including x1 (raw), NOT_CONFIGURED, and FLOAT's x4 (Lucene 7-bit) and x2 (Faiss fp16),
         * which are stored in other formats.
         */
        public static boolean isSQCoded(CompressionLevel compressionLevel, VectorDataType vectorDataType) {
            // Only float and half_float have an SQ path; for other types the arithmetic below would be
            // meaningless (binary at x1 computes to 1 bit).
            if (vectorDataType != VectorDataType.FLOAT && vectorDataType != VectorDataType.HALF_FLOAT) {
                return false;
            }
            // fromValue falls back to FULL_PRECISION for widths with no constant, which is never SQ-coded.
            return CompressionLevel.isConfigured(compressionLevel)
                && SQ_CODED_BITS.contains(fromValue(vectorDataType.getCompressionBits(compressionLevel)));
        }

        /**
         * Compression this bit width achieves for {@code vectorDataType}. The constants above are
         * measured against FLOAT's 32 bits, so {@link #ONE} is x32 there; HALF_FLOAT's 16 bits are half
         * that, so the same bit widths land one compression level lower.
         *
         * HALF_FLOAT supports only bits ∈ (1, 2, 4). Any other width is rejected.
         */
        public CompressionLevel getCompressionLevel(VectorDataType vectorDataType) {
            if (vectorDataType == VectorDataType.HALF_FLOAT) {
                if (SQ_CODED_BITS.contains(this) == false) {
                    throw new IllegalArgumentException(
                        String.format(Locale.ROOT, "half_float only supports bits in {1, 2, 4} for SQ quantization, got bits=%d", value)
                    );
                }
                return CompressionLevel.fromFactor(vectorDataType.getBitsPerDimension() / value);
            }
            return compressionLevel;
        }

        /**
         * Data-type-aware inverse of {@link #getCompressionLevel(VectorDataType)}. For HALF_FLOAT,
         * x16/x8/x4 are its SQ 1/2/4-bit levels; any other compression level (including x1) falls
         * through to the generic, non-data-type-aware mapping below.
         */
        public static QuantizationBits fromCompressionLevel(CompressionLevel compressionLevel, VectorDataType vectorDataType) {
            if (vectorDataType == VectorDataType.HALF_FLOAT && isSQCoded(compressionLevel, vectorDataType)) {
                return fromValue(vectorDataType.getCompressionBits(compressionLevel));
            }
            return fromCompressionLevel(compressionLevel);
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
     * Validates encoder configuration for the create-index path.
     * Throws ValidationException if the configuration is invalid.
     * Training-specific validation remains in {@link #validateEncoderConfig(TrainingConfigValidationInput)}.
     *
     * @param resolvedMethodContext the resolved method context
     * @param configContext method config context
     * @throws ValidationException if validation fails
     */
    default void validate(KNNMethodContext resolvedMethodContext, KNNMethodConfigContext configContext) {
        // no-op by default — encoders without validation constraints inherit this
    }

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

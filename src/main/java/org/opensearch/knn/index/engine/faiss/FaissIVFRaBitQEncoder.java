/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import com.google.common.collect.ImmutableSet;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.common.featureflags.KNNFeatureFlags;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.engine.TrainingConfigValidationInput;
import org.opensearch.knn.index.engine.TrainingConfigValidationOutput;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.EnumSet;
import java.util.Locale;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.BYTES_PER_KILOBYTES;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_RABITQ_QUERY_BITS;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_RABITQ_QUERY_BITS_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_RABITQ_QUERY_BITS_MAX;
import static org.opensearch.knn.common.KNNConstants.FAISS_RABITQ_DESCRIPTION;

/**
 * Faiss IVF RaBitQ encoder. Builds a Faiss {@code IndexIVFRaBitQ} behind a random rotation
 * which encodes each rotated vector's residual against its IVF centroid as 1 bit per dimension plus two float
 * correction factors, and ranks candidates with RaBitQ's distance estimator. Supported for L2 and inner product (and so
 * cosine).
 *
 * <p>{@code query_bits} is the number of bits the query is scalar quantized to (Faiss {@code qb}). With 1 to 8 bits the
 * estimator runs on popcounts, one pass per bit; 0 keeps the query in float and scores bit by bit, which is much slower.
 * The default of 4 keeps the recall of 8 bits at a lower scan cost; fewer bits lose recall.
 */
public class FaissIVFRaBitQEncoder implements Encoder {

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT);

    /**
     * Per-vector correction factors Faiss stores after the sign bits (2 floats, see faiss::RaBitQuantizer).
     */
    static final int FACTORS_BITS = 2 * Float.SIZE;

    static final String DISABLED_ERROR_MESSAGE = String.format(
        Locale.ROOT,
        "Encoder [%s] is experimental and disabled. Set cluster setting [%s] to true to enable it.",
        KNNConstants.ENCODER_RABITQ,
        KNNFeatureFlags.KNN_RABITQ_ENABLED_SETTING.getKey()
    );

    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(KNNConstants.ENCODER_RABITQ)
        .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
        .addParameter(
            ENCODER_PARAMETER_RABITQ_QUERY_BITS,
            new Parameter.IntegerParameter(
                ENCODER_PARAMETER_RABITQ_QUERY_BITS,
                ENCODER_PARAMETER_RABITQ_QUERY_BITS_DEFAULT,
                (v, context) -> v >= 0 && v <= ENCODER_PARAMETER_RABITQ_QUERY_BITS_MAX
            )
        )
        .setRequiresTraining(true)
        // The random rotation is a dense d x d float matrix: (4 * d * d) / 1024 + 1
        .setOverheadInKBEstimator(
            (methodComponent, methodComponentContext, dimension) -> ((long) Float.BYTES * dimension * dimension) / BYTES_PER_KILOBYTES + 1
        )
        .setKnnLibraryIndexingContextGenerator(
            ((methodComponent, methodComponentContext, knnMethodConfigContext) -> MethodAsMapBuilder.builder(
                FAISS_RABITQ_DESCRIPTION,
                methodComponent,
                methodComponentContext,
                knnMethodConfigContext
            ).build())
        )
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
        if (knnMethodConfigContext == null || knnMethodConfigContext.getDimension() == null) {
            return CompressionLevel.NOT_CONFIGURED;
        }
        // A d-dimensional float32 vector goes from 32 * d bits to d sign bits plus the correction factors. Round
        // down to the largest supported level the encoding actually achieves, e.g. d=768 -> 29.5x -> x16.
        int dimension = knnMethodConfigContext.getDimension();
        // The ratio is always below 32, so the largest power of two at or under it is a defined level.
        int actualCompression = (dimension * Float.SIZE) / (dimension + FACTORS_BITS);
        return CompressionLevel.fromFactor(Integer.highestOneBit(Math.max(1, actualCompression)));
    }

    /**
     * RaBitQ is experimental: reject new configurations (create-index and train-model resolution) unless
     * {@link KNNFeatureFlags#KNN_RABITQ_ENABLED_SETTING} is enabled. Indices built from an already trained model do not
     * resolve the method again, so they keep working when the flag is turned off.
     */
    @Override
    public void validate(KNNMethodContext resolvedMethodContext, KNNMethodConfigContext configContext) {
        if (KNNFeatureFlags.isRaBitQEnabled() == false) {
            ValidationException validationException = new ValidationException();
            validationException.addValidationError(DISABLED_ERROR_MESSAGE);
            throw validationException;
        }
    }

    @Override
    public TrainingConfigValidationOutput validateEncoderConfig(TrainingConfigValidationInput validationInput) {
        if (KNNFeatureFlags.isRaBitQEnabled() == false) {
            return TrainingConfigValidationOutput.builder().valid(false).errorMessage(DISABLED_ERROR_MESSAGE).build();
        }
        return TrainingConfigValidationOutput.builder().build();
    }

    @Override
    public EncoderType getEncoderType() {
        return EncoderType.RABITQ;
    }

    @Override
    public Set<QuantizationBits> getSupportedBits() {
        return EnumSet.of(QuantizationBits.ONE);
    }

    @Override
    public QuantizationBits getQuantizationBits() {
        return QuantizationBits.ONE;
    }
}

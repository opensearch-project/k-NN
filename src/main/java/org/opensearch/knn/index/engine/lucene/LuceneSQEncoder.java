/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import com.google.common.collect.ImmutableSet;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

import org.opensearch.Version;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Arrays;
import java.util.Locale;
import java.util.Set;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.DYNAMIC_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.MAXIMUM_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.MINIMUM_CONFIDENCE_INTERVAL;

/**
 * Lucene scalar quantization encoder
 */
public class LuceneSQEncoder implements Encoder {
    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT);
    static final Set<Integer> LUCENE_SQ_BITS_SUPPORTED = Arrays.stream(Bits.values())
        .map(Bits::getValue)
        .collect(Collectors.toUnmodifiableSet());
    static final Bits LUCENE_PRE_360_SUPPORTED_SQ_BITS = Bits.SEVEN;

    /**
     * Supported bit widths for SQ quantization. Each maps to a specific quantization strategy
     * and compression level.
     */
    @Getter
    @RequiredArgsConstructor
    public enum Bits {
        ONE(1, CompressionLevel.x32),
        SEVEN(7, CompressionLevel.x4);

        private final int value;
        private final CompressionLevel compressionLevel;

        public static Bits fromValue(int value) {
            for (Bits b : values()) {
                if (b.value == value) return b;
            }
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Unsupported bits value: %d", value));
        }
    }

    // Lucene SQ supports compression to 1 bit only in indices with version >= 3.6.0
    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(ENCODER_SQ)
        .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
        .addParameter(
            LUCENE_SQ_CONFIDENCE_INTERVAL,
            new Parameter.DoubleParameter(
                LUCENE_SQ_CONFIDENCE_INTERVAL,
                null,
                (v, context) -> v == DYNAMIC_CONFIDENCE_INTERVAL || (v >= MINIMUM_CONFIDENCE_INTERVAL && v <= MAXIMUM_CONFIDENCE_INTERVAL)
            )
        )
        .addParameter(
            LUCENE_SQ_BITS,
            // Making default value null - it should be passed in from LuceneHNSWMethodResolver
            new Parameter.IntegerParameter(LUCENE_SQ_BITS, null, (v, context) -> LUCENE_SQ_BITS_SUPPORTED.contains(v))
        )
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
        if (knnMethodConfigContext == null) {
            return CompressionLevel.x4;
        }

        if (CompressionLevel.isConfigured(knnMethodConfigContext.getCompressionLevel())) {
            return knnMethodConfigContext.getCompressionLevel();
        }

        // resolve compression level based on bits if its specified - the two must be equivalent
        if (methodComponentContext != null && methodComponentContext.getParameters() != null) {
            Object bitsObj = methodComponentContext.getParameters().get(LUCENE_SQ_BITS);
            if (bitsObj instanceof Integer) {
                return Bits.fromValue((Integer) bitsObj).getCompressionLevel();
            }
        }

        // For indices after version 3.6.0, we want to default to 32x compression
        if (knnMethodConfigContext.getVersionCreated().onOrAfter(Version.V_3_6_0)) {
            return CompressionLevel.x32;
        }
        return CompressionLevel.x4;
    }
}

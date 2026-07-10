/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.Builder;
import lombok.Getter;
import org.opensearch.Version;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

/**
 * Immutable value object holding resolved index configuration.
 * Constructed once at resolution time, read by both write and query paths.
 * All behavioral decisions are methods derived from stored primitives.
 */
@Builder
@Getter
public final class ResolvedIndexSpec {
    private final KNNEngine engine;
    private final String methodName;
    private final Encoder.EncoderType encoderType;
    private final Encoder.QuantizationBits quantizationBits;
    @Builder.Default
    private final CompressionLevel compressionLevel = CompressionLevel.NOT_CONFIGURED;
    @Builder.Default
    private final Mode mode = Mode.NOT_CONFIGURED;
    private final VectorDataType vectorDataType;
    private final int dimension;
    private final Version indexVersionCreated;

    /**
     * Whether this configuration always uses memory optimized search.
     */
    public boolean alwaysUseMemoryOptimizedSearch() {
        return isSQOneBit();
    }

    /**
     * Whether this configuration is eligible for memory optimized search.
     * Faiss HNSW with FLAT, SQ, or BQ encoders.
     */
    public boolean isMemoryOptimizedEligible() {
        return engine == KNNEngine.FAISS
            && METHOD_HNSW.equals(methodName)
            && (encoderType == Encoder.EncoderType.FLAT || encoderType == Encoder.EncoderType.SQ || encoderType == Encoder.EncoderType.BQ);
    }

    /**
     * Whether this configuration supports radial search.
     *
     * <p>Radial search is blocked for:</p>
     * <ul>
     *   <li>Engines that do not support radial search (NMSLIB)</li>
     *   <li>Binary vector data type</li>
     *   <li>BQ (binary quantization) encoder</li>
     *   <li>Quantized indices that are not 1-bit SQ — among quantized indices, only the
     *       flat method or SQ encoder with bits=1 support radial search via rescoring</li>
     * </ul>
     */
    public boolean supportsRadialSearch() {
        if (KNNEngine.ENGINES_SUPPORTING_RADIAL_SEARCH.contains(engine) == false) {
            return false;
        }
        if (vectorDataType == VectorDataType.BINARY) {
            return false;
        }
        if (encoderType == Encoder.EncoderType.BQ) {
            return false;
        }
        if (isQuantizedIndex()) {
            return isMethodFlat() || isSQOneBit();
        }
        return true;
    }

    /**
     * Returns the appropriate rescore context for this configuration.
     * Recomputed each call since RescoreContext has mutable state in getFirstPassK().
     */
    public RescoreContext getRescoreContext() {
        return compressionLevel.getDefaultRescoreContext(
            mode,
            dimension,
            indexVersionCreated != null ? indexVersionCreated : Version.CURRENT,
            isMethodFlat(),
            isSQOneBit(),
            engine
        );
    }

    /**
     * Whether rescoring is needed for this configuration.
     */
    public boolean requiresRescore() {
        return getRescoreContext() != null;
    }

    /**
     * Whether this is an SQ encoder producing fp16 quantized output. Distinct from native halfFloat data type storage.
     * Equivalent to FaissHNSWMethod.isFloat16Index:
     * encoder is SQ, dataType is FLOAT, and bits is SIXTEEN or legacy (null/FULL_PRECISION defaults to fp16).
     */
    public boolean isFP16QuantizedIndex() {
        // FULL_PRECISION represents the legacy path where SQ encoder is specified without explicit bits param.
        // buildResolvedIndexSpec defaults to FULL_PRECISION in this case, and the old FaissHNSWMethod.isFloat16Index()
        // treated SQ-with-no-bits as fp16 by convention.
        return vectorDataType == VectorDataType.FLOAT
            && encoderType == Encoder.EncoderType.SQ
            && (quantizationBits == Encoder.QuantizationBits.SIXTEEN || quantizationBits == Encoder.QuantizationBits.FULL_PRECISION);
    }

    public boolean isSQOneBit() {
        return encoderType == Encoder.EncoderType.SQ && quantizationBits == Encoder.QuantizationBits.ONE;
    }

    public boolean isFaissSQOneBit() {
        return engine == KNNEngine.FAISS && isSQOneBit();
    }

    private boolean isMethodFlat() {
        return METHOD_FLAT.equals(methodName);
    }

    /**
     * Returns true when the configured compression level implies lossy quantization that needs special handling.
     * x1 is full precision and x2 is fp16 (near-lossless), so both are excluded; x4 and above are lossy.
     */
    private boolean isQuantizedIndex() {
        return CompressionLevel.isConfigured(compressionLevel)
            && compressionLevel != CompressionLevel.x1
            && compressionLevel != CompressionLevel.x2;
    }
}

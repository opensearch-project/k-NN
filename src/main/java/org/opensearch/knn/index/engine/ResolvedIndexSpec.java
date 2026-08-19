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
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;

/**
 * Immutable value object holding resolved index configuration.
 * Constructed once at resolution time, read by both write and query paths.
 * All behavioral decisions are methods derived from stored primitives.
 */
@Builder(toBuilder = true)
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
     * Whether this spec was resolved from model metadata rather than an explicit method mapping.
     * Model-derived specs keep parity with the legacy model-path validation, which never applied
     * compression-level restrictions (see {@link #supportsRadialSearch()}).
     */
    @Builder.Default
    private final boolean modelBased = false;

    /**
     * Creates a spec for a field with no ANN structure: flat (index.knn=false) fields and
     * model fields whose metadata predates method serialization. All behavioral methods
     * return "off" answers — no radial search, no default rescore, no memory-optimized search,
     * no remote index build.
     *
     * @param vectorDataType data type of the vector field
     * @param dimension dimension of the vector field
     * @param indexVersionCreated version the index was created on
     * @return a spec representing "no ANN configuration"
     */
    public static ResolvedIndexSpec noAnn(VectorDataType vectorDataType, int dimension, Version indexVersionCreated) {
        return ResolvedIndexSpec.builder()
            .vectorDataType(vectorDataType)
            .dimension(dimension)
            .indexVersionCreated(indexVersionCreated)
            .build();
    }

    /**
     * Creates a spec for a model field whose metadata predates method serialization.
     * Same behavioral defaults as {@link #noAnn} but retains the model engine and marks
     * the spec as model-based for radial search gating.
     */
    public static ResolvedIndexSpec noAnnFromModel(
        KNNEngine engine,
        VectorDataType vectorDataType,
        int dimension,
        Version indexVersionCreated
    ) {
        return ResolvedIndexSpec.builder()
            .engine(engine)
            .vectorDataType(vectorDataType)
            .dimension(dimension)
            .indexVersionCreated(indexVersionCreated)
            .modelBased(true)
            .build();
    }

    /**
     * Returns a copy of this spec marked as model-derived.
     * Used when a fully resolved spec (from method resolution) needs the model-based
     * flag for radial search gating in {@link #supportsRadialSearch()}.
     */
    public ResolvedIndexSpec asModelBased() {
        return toBuilder().modelBased(true).build();
    }

    /**
     * Whether this configuration always uses memory optimized search.
     * SQ 1-bit indices require memory optimized search for correctness, regardless of engine
     * (both Faiss and Lucene SQ 1-bit rely on it). The one exclusion is IVF: IVF-based SQ 1-bit
     * models (e.g. trained with on_disk/32x) produce IVF Faiss indices that the memory optimized
     * reader cannot load, so forcing memory optimized search for them would fail at query time.
     */
    public boolean alwaysUseMemoryOptimizedSearch() {
        return isSQOneBit() && METHOD_IVF.equals(methodName) == false;
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
     *   <li>All quantized indices (compression level other than {@code x1}/{@code x2}), regardless of
     *       encoder or method. Radial search was disabled for quantized indices due to low recall
     *       (see <a href="https://github.com/opensearch-project/k-NN/pull/3464">#3464</a>); there are
     *       no longer any exceptions for flat method or 1-bit SQ.</li>
     * </ul>
     *
     * <p>Model-derived specs skip the compression-level restriction: the legacy model-path
     * validation only blocked BQ (via {@code QuantizationConfig != EMPTY}), so quantized
     * models such as PQ/IVF-PQ remain allowed.</p>
     */
    public boolean supportsRadialSearch() {
        if (engine == null || engine.supportsRadialSearch() == false) {
            return false;
        }
        if (vectorDataType == VectorDataType.BINARY) {
            return false;
        }
        if (encoderType == Encoder.EncoderType.BQ) {
            return false;
        }
        // All quantized indices block radial search — no flat-method or 1-bit SQ exception (#3464).
        if (modelBased == false && isQuantizedIndex()) {
            return false;
        }
        return true;
    }

    /**
     * Whether this configuration requires memory-optimized search regardless of cluster setting.
     * True for ON_DISK mode with x1 (no) compression — these indices store vectors on disk and
     * need memory-optimized search for acceptable latency.
     */
    public boolean requiresMemoryOptimizedSearchForOnDisk() {
        return mode == Mode.ON_DISK && compressionLevel == CompressionLevel.x1;
    }

    private static final float FLAT_OVERSAMPLE_FACTOR = 2.0f;

    /**
     * Returns the default rescore context for this configuration, or null when no rescoring applies.
     *
     * <p>Resolution order:</p>
     * <ol>
     *   <li>SQ 1-bit: fixed 2x oversample (quantized distances need full-precision rescoring)</li>
     *   <li>x32 compression with the flat method: 2x oversample</li>
     *   <li>Otherwise, only when the compression level requires rescoring for the resolved mode
     *       ({@link CompressionLevel#isModeValidForRescore}):
     *     <ul>
     *       <li>x32 + Lucene engine on indices created on/after 3.6: Lucene scalar quantizer default</li>
     *       <li>x4 on indices created before 3.1: no rescoring (BWC)</li>
     *       <li>dimension at or below {@link RescoreContext#DIMENSION_THRESHOLD} (except x4): 5x oversample</li>
     *       <li>otherwise the compression level own default</li>
     *     </ul>
     *   </li>
     *   <li>All other configurations: null (no default rescore)</li>
     * </ol>
     *
     * <p>Recomputed on each call since {@link RescoreContext} carries mutable state in
     * {@code getFirstPassK()}.</p>
     */
    public RescoreContext getRescoreContext() {
        if (isSQOneBit()) {
            return RescoreContext.builder()
                .oversampleFactor(RescoreContext.FAISS_SCALAR_QUANTIZED_INDEX_OVERSAMPLE_FACTOR)
                .allowOverrideOversampleFactor(false)
                .userProvided(false)
                .build();
        }

        if (compressionLevel == CompressionLevel.x32 && isMethodFlat()) {
            return RescoreContext.builder().oversampleFactor(FLAT_OVERSAMPLE_FACTOR).userProvided(false).build();
        }

        if (compressionLevel.isModeValidForRescore(mode)) {
            Version version = indexVersionCreated != null ? indexVersionCreated : Version.CURRENT;

            if (compressionLevel == CompressionLevel.x32 && engine == KNNEngine.LUCENE && version.onOrAfter(Version.V_3_6_0)) {
                return RescoreContext.builder()
                    .oversampleFactor(RescoreContext.OVERSAMPLE_FACTOR_DEFAULT_FOR_LUCENE_SCALAR_QUANTIZER_AFTER_V360)
                    .userProvided(false)
                    .build();
            }

            if (compressionLevel == CompressionLevel.x4 && version.before(Version.V_3_1_0)) {
                return null;
            }

            if (compressionLevel != CompressionLevel.x4 && dimension <= RescoreContext.DIMENSION_THRESHOLD) {
                return RescoreContext.builder()
                    .oversampleFactor(RescoreContext.OVERSAMPLE_FACTOR_BELOW_DIMENSION_THRESHOLD)
                    .userProvided(false)
                    .build();
            }

            return compressionLevel.getDefaultRescoreContext();
        }

        return null;
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

    /**
     * Whether this configuration supports remote index build.
     *
     * <p>Remote build requires the Faiss engine and the HNSW method. Within that, the following
     * configurations are supported (mirroring what the remote build service can build):</p>
     * <ul>
     *   <li>SQ 1-bit (binary quantization via scalar quantizer)</li>
     *   <li>FP16 (SQ encoder with bits=16)</li>
     *   <li>FLOAT with FLAT encoder (full precision float32)</li>
     *   <li>BINARY with FLAT encoder (user-provided binary vectors)</li>
     *   <li>BYTE with FLAT encoder</li>
     *   <li>FLOAT with SQ encoder at intermediate bit widths (2, 4, 7 bits)</li>
     * </ul>
     *
     * <p>Everything else (e.g. PQ encoder, non-FLAT binary/byte configs) returns false. New encoder
     * types must be validated against the remote build service before being added here.</p>
     */
    public boolean supportsRemoteIndexBuild() {
        if (engine != KNNEngine.FAISS || !METHOD_HNSW.equals(methodName)) {
            return false;
        }

        if (isSQOneBit() || isFP16QuantizedIndex()) {
            return true;
        }

        if (encoderType == Encoder.EncoderType.FLAT) {
            return vectorDataType == VectorDataType.FLOAT
                || vectorDataType == VectorDataType.BINARY
                || vectorDataType == VectorDataType.BYTE;
        }

        if (vectorDataType == VectorDataType.FLOAT
            && encoderType == Encoder.EncoderType.SQ
            && quantizationBits != null
            && quantizationBits != Encoder.QuantizationBits.ONE
            && quantizationBits != Encoder.QuantizationBits.SIXTEEN
            && quantizationBits != Encoder.QuantizationBits.FULL_PRECISION) {
            return true;
        }

        return false;
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

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

/**
 * SVS-specific constants used by the sandbox SVS engine classes.
 *
 * <p>These constants are intentionally kept local to the sandbox module so that the
 * experimental SVS integration does not leak names into the main {@code KNNConstants}.
 * Constants that already exist in the main module (e.g. {@code ENCODER_FLAT},
 * {@code METHOD_ENCODER_PARAMETER}) are imported from {@code KNNConstants} directly.</p>
 */
public final class SVSConstants {

    private SVSConstants() {}

    // The engine name users specify (engine: "svs") and the on-disk file extension for SVS index files.
    // KNNEngine.EXPERIMENTAL adopts this name/extension at runtime via the SandboxEngineProvider.
    public static final String SVS_ENGINE_NAME = "svs";
    public static final String SVS_EXTENSION = ".svs";

    // Method names
    public static final String METHOD_SVS_VAMANA = "svs_vamana";

    // Faiss index-factory description prefixes
    public static final String FAISS_SVS_VAMANA_DESCRIPTION = "SVSVamana";

    // Encoder names
    // FP16 and SQ8 are exposed through the unified {@code sq} encoder (imported from
    // {@link org.opensearch.knn.common.KNNConstants#ENCODER_SQ}) via the {@code type} parameter below.
    public static final String FAISS_SVS_ENCODER_LVQ = "lvq";

    // sq encoder type parameter (mirrors the HNSW sq encoder's {@code type} convention).
    public static final String FAISS_SVS_SQ_TYPE = "type";
    public static final String FAISS_SVS_SQ_TYPE_FP16 = "fp16";
    public static final String FAISS_SVS_SQ_TYPE_SQ8 = "sq8";

    // Faiss index-factory descriptions emitted per sq type.
    public static final String FAISS_SVS_SQ_FP16_DESCRIPTION = "FP16";
    public static final String FAISS_SVS_SQ_SQ8_DESCRIPTION = "SQI8";

    // Vamana method parameters
    public static final String METHOD_PARAMETER_DEGREE = "degree";
    public static final String METHOD_PARAMETER_CONSTRUCTION_WINDOW_SIZE = "construction_window_size";
    public static final String METHOD_PARAMETER_SEARCH_WINDOW_SIZE = "search_window_size";
    public static final String METHOD_PARAMETER_SEARCH_BUFFER_CAPACITY = "search_buffer_capacity";
    public static final String METHOD_PARAMETER_ALPHA = "alpha";

    // LVQ encoder parameters
    public static final String METHOD_PARAMETER_LVQ_PRIMARY_BITS = "primary_bits";
    public static final String METHOD_PARAMETER_LVQ_RESIDUAL_BITS = "residual_bits";

    // Default construction window size (mirrors main KNNSettings default in 3.4-era code).
    public static final int DEFAULT_CONSTRUCTION_WINDOW_SIZE = 128;
}

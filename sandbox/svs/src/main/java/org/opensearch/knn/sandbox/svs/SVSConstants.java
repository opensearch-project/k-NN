/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.engine.ParameterKey;

/**
 * SVS tenant constants; names that already exist in core {@code KNNConstants} are imported from there.
 */
public final class SVSConstants {

    private SVSConstants() {}

    public static final String SVS_ENGINE_NAME = "svs";
    public static final String SVS_EXTENSION = ".svs";

    public static final String METHOD_SVS_VAMANA = "svs_vamana";

    public static final String FAISS_SVS_VAMANA_DESCRIPTION = "SVSVamana";

    public static final String FAISS_SVS_ENCODER_LVQ = "lvq";

    public static final String FAISS_SVS_SQ_TYPE = "type";
    public static final String FAISS_SVS_SQ_TYPE_FP16 = "fp16";
    public static final String FAISS_SVS_SQ_TYPE_SQ8 = "sq8";

    // Faiss renamed the SQI8 factory token to SQ8 (facebookresearch/faiss#5337).
    public static final String FAISS_SVS_SQ_FP16_DESCRIPTION = "FP16";
    public static final String FAISS_SVS_SQ_SQ8_DESCRIPTION = "SQ8";

    public static final String METHOD_PARAMETER_DEGREE = "degree";
    public static final String METHOD_PARAMETER_CONSTRUCTION_WINDOW_SIZE = "construction_window_size";
    public static final String METHOD_PARAMETER_SEARCH_WINDOW_SIZE = "search_window_size";
    public static final String METHOD_PARAMETER_SEARCH_BUFFER_CAPACITY = "search_buffer_capacity";
    public static final String METHOD_PARAMETER_ALPHA = "alpha";

    public static final String METHOD_PARAMETER_LVQ_PRIMARY_BITS = "primary_bits";
    public static final String METHOD_PARAMETER_LVQ_RESIDUAL_BITS = "residual_bits";

    public static final String FAISS_SVS_ENCODER_LEANVEC = "leanvec";
    public static final String METHOD_PARAMETER_LEANVEC_DIMENSIONS = "dimensions";
    public static final String METHOD_PARAMETER_LEANVEC_TRAINING_THRESHOLD = "training_threshold";
    public static final String METHOD_PARAMETER_LEANVEC_ROUGH_TRAINING_THRESHOLD = "rough_training_threshold";
    public static final int LEANVEC_DEFAULT_TRAINING_THRESHOLD = 100_000;
    public static final int LEANVEC_DEFAULT_ROUGH_TRAINING_THRESHOLD = 10_000;

    public static final int DEFAULT_CONSTRUCTION_WINDOW_SIZE = 128;

    public static final ParameterKey<Integer> INDEX_THREAD_QTY_KEY = ParameterKey.intKey(KNNConstants.INDEX_THREAD_QTY);
}

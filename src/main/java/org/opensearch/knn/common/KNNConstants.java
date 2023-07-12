/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

import org.opensearch.knn.index.VectorDataType;

public class KNNConstants {
    // shared across library constants
    public static final String DIMENSION = "dimension";
    public static final String KNN_ENGINE = "engine";
    public static final String KNN_METHOD = "method";
    public static final String NAME = "name";
    public static final String PARAMETERS = "parameters";
    public static final String METHOD_HNSW = "hnsw";
    public static final String METHOD_PARAMETER_EF_SEARCH = "ef_search";
    public static final String METHOD_PARAMETER_EF_CONSTRUCTION = "ef_construction";
    public static final String METHOD_PARAMETER_M = "m";
    public static final String METHOD_IVF = "ivf";
    public static final String METHOD_PARAMETER_NLIST = "nlist";
    public static final String METHOD_PARAMETER_SPACE_TYPE = "space_type"; // used for mapping parameter
    public static final String COMPOUND_EXTENSION = "c";
    public static final String MODEL = "model";
    public static final String MODELS = "models";
    public static final String MODEL_ID = "model_id";
    public static final String MODEL_BLOB_PARAMETER = "model_blob";
    public static final String MODEL_INDEX_MAPPING_PATH = "mappings/model-index.json";
    public static final String MODEL_INDEX_NAME = ".opensearch-knn-models";
    public static final String PLUGIN_NAME = "knn";
    public static final String MODEL_METADATA_FIELD = "knn-models";
    public static final Integer BYTES_PER_KILOBYTES = 1024;
    public static final String PREFERENCE_PARAMETER = "preference";

    public static final String MODEL_STATE = "state";
    public static final String MODEL_TIMESTAMP = "timestamp";
    public static final String MODEL_DESCRIPTION = "description";
    public static final String MODEL_ERROR = "error";
    public static final String PARAM_SIZE = "size";
    public static final Integer SEARCH_MODEL_MIN_SIZE = 1;
    public static final Integer SEARCH_MODEL_MAX_SIZE = 1000;

    public static final String KNN_THREAD_POOL_PREFIX = "knn";
    public static final String TRAIN_THREAD_POOL = "training";

    public static final String TRAINING_JOB_COUNT_FIELD_NAME = "training_job_count";
    public static final String NODES_KEY = "nodes";

    public static final String TRAIN_INDEX_PARAMETER = "training_index";
    public static final String TRAIN_FIELD_PARAMETER = "training_field";
    public static final String MAX_VECTOR_COUNT_PARAMETER = "max_training_vector_count";
    public static final String SEARCH_SIZE_PARAMETER = "search_size";

    public static final String VECTOR_DATA_TYPE_FIELD = "data_type";
    public static final VectorDataType DEFAULT_VECTOR_DATA_TYPE_FIELD = VectorDataType.FLOAT;

    // Lucene specific constants
    public static final String LUCENE_NAME = "lucene";

    // nmslib specific constants
    public static final String NMSLIB_NAME = "nmslib";
    public static final String SPACE_TYPE = "spaceType"; // used as field info key
    public static final String HNSW_ALGO_M = "M";
    public static final String HNSW_ALGO_EF_CONSTRUCTION = "efConstruction";
    public static final String HNSW_ALGO_EF_SEARCH = "efSearch";
    public static final String INDEX_THREAD_QTY = "indexThreadQty";

    // Faiss specific constants
    public static final String FAISS_NAME = "faiss";
    public final static String FAISS_EXTENSION = ".faiss";
    public static final String INDEX_DESCRIPTION_PARAMETER = "index_description";
    public static final String METHOD_ENCODER_PARAMETER = "encoder";
    public static final String METHOD_PARAMETER_NPROBES = "nprobes";
    public static final String ENCODER_FLAT = "flat";
    public static final String ENCODER_PQ = "pq";
    public static final String ENCODER_PARAMETER_PQ_M = "m";
    public static final String ENCODER_PARAMETER_PQ_CODE_SIZE = "code_size";
    public static final String FAISS_HNSW_DESCRIPTION = "HNSW";
    public static final String FAISS_IVF_DESCRIPTION = "IVF";
    public static final String FAISS_FLAT_DESCRIPTION = "Flat";
    public static final String FAISS_PQ_DESCRIPTION = "PQ";

    // Parameter defaults/limits
    public static final Integer ENCODER_PARAMETER_PQ_CODE_COUNT_DEFAULT = 1;
    public static final Integer ENCODER_PARAMETER_PQ_CODE_COUNT_LIMIT = 1024;
    public static final Integer ENCODER_PARAMETER_PQ_CODE_SIZE_DEFAULT = 8;
    public static final Integer ENCODER_PARAMETER_PQ_CODE_SIZE_LIMIT = 128;
    public static final Integer METHOD_PARAMETER_NLIST_DEFAULT = 4;
    public static final Integer METHOD_PARAMETER_NPROBES_DEFAULT = 1;
    public static final Integer METHOD_PARAMETER_NPROBES_LIMIT = 20000;
    public static final Integer METHOD_PARAMETER_NLIST_LIMIT = 20000;
    public static final Integer MAX_MODEL_DESCRIPTION_LENGTH = 1000; // max number of chars a model's description can be
    public static final Integer MODEL_CACHE_CAPACITY_ATROPHY_THRESHOLD_IN_MINUTES = 30;
    public static final Integer MODEL_CACHE_EXPIRE_AFTER_ACCESS_TIME_MINUTES = 30;

    // Lib names
    private static final String JNI_LIBRARY_PREFIX = "opensearchknn_";
    public static final String FAISS_JNI_LIBRARY_NAME = JNI_LIBRARY_PREFIX + FAISS_NAME;
    public static final String NMSLIB_JNI_LIBRARY_NAME = JNI_LIBRARY_PREFIX + NMSLIB_NAME;
}

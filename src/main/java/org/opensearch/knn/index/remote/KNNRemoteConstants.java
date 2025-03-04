/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

// Public class to define the constants used by Remote Index Build in 2 or more classes.
public class KNNRemoteConstants {
    // Repository filepath constants
    public static final String VECTOR_BLOB_FILE_EXTENSION = ".knnvec";
    public static final String DOC_ID_FILE_EXTENSION = ".knndid";
    public static final String VECTORS_PATH = "_vectors";

    // Repository-S3
    public static final String S3 = "s3";
    public static final String BUCKET = "bucket";

    // Build request keys
    public static final String ALGORITHM = "algorithm";
    public static final String ALGORITHM_PARAMETERS = "algorithm_parameters";
    public static final String INDEX_PARAMETERS = "index_parameters";
    public static final String DOC_COUNT = "doc_count";
    public static final String TENANT_ID = "tenant_id";
    public static final String DOC_ID_PATH = "doc_id_path";
    public static final String VECTOR_PATH = "vector_path";
    public static final String CONTAINER_NAME = "container_name";
    public static final String REPOSITORY_TYPE = "repository_type";

    // HTTP implementation
    public static final String BASIC_PREFIX = "Basic ";
    public static final String BUILD_ENDPOINT = "/_build";
    public static final String STATUS_ENDPOINT = "/_status";

    // Status response keys
    public static final String TASK_STATUS = "task_status";
    public static final String RUNNING_INDEX_BUILD = "RUNNING_INDEX_BUILD";
    public static final String COMPLETED_INDEX_BUILD = "COMPLETED_INDEX_BUILD";
    public static final String FAILED_INDEX_BUILD = "FAILED_INDEX_BUILD";
    public static final String FILE_NAME = "file_name";
    public static final String ERROR_MESSAGE = "error_message";
    public static final String MOCK_FILE_NAME = "graph.faiss";
}

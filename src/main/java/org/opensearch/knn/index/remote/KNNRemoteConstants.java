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

    // HTTP implementation
    public static final String BUILD_ENDPOINT = "/_build";
    public static final String STATUS_ENDPOINT = "/_status";
}

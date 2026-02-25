/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.remoteindexbuild.model;

import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.remoteindexbuild.client.RemoteIndexHTTPClientTests;
import org.opensearch.test.OpenSearchSingleNodeTestCase;

import java.io.IOException;
import java.util.Map;

import static org.opensearch.remoteindexbuild.TestConstants.FAISS;
import static org.opensearch.remoteindexbuild.TestConstants.FLOAT;
import static org.opensearch.remoteindexbuild.TestConstants.HNSW_ALGORITHM;
import static org.opensearch.remoteindexbuild.TestConstants.L2_SPACE_TYPE;
import static org.opensearch.remoteindexbuild.TestConstants.TEST_BUCKET;
import static org.opensearch.remoteindexbuild.TestConstants.TEST_CLUSTER;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.ALGORITHM;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.ALGORITHM_PARAMETERS;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.CONTAINER_NAME;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.DIMENSION;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.DOC_COUNT;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.DOC_ID_FILE_EXTENSION;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.DOC_ID_PATH;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.INDEX_PARAMETERS;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.KNN_ENGINE;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.METHOD_PARAMETER_M;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.REPOSITORY_TYPE;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.S3;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.TENANT_ID;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.VECTOR_BLOB_FILE_EXTENSION;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.VECTOR_PATH;

public class RemoteBuildRequestTests extends OpenSearchSingleNodeTestCase {
    public static final String MOCK_FULL_PATH = "vectors/1_1_25/SIRKos4rOWlMA62PX2p75m_vectors/SIRKos4rOWlMA62PX2p75m_target_field__3l";

    public void testToXContent() throws IOException {
        RemoteBuildRequest request = RemoteBuildRequest.builder()
            .repositoryType(S3)
            .containerName(TEST_BUCKET)
            .vectorPath(MOCK_FULL_PATH + VECTOR_BLOB_FILE_EXTENSION)
            .docIdPath(MOCK_FULL_PATH + DOC_ID_FILE_EXTENSION)
            .tenantId(TEST_CLUSTER)
            .dimension(2)
            .docCount(2)
            .vectorDataType(FLOAT)
            .engine(FAISS)
            .indexParameters(
                RemoteFaissHNSWIndexParameters.builder()
                    .algorithm(HNSW_ALGORITHM)
                    .spaceType(L2_SPACE_TYPE)
                    .efConstruction(94)
                    .efSearch(89)
                    .m(14)
                    .build()
            )
            .build();

        String expectedJson = getMockExpectedJson();

        // Use JSON parser to compare trees because order is not guaranteed
        XContentParser expectedParser = JsonXContent.jsonXContent.createParser(
            NamedXContentRegistry.EMPTY,
            DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
            expectedJson
        );
        Map<String, Object> expectedMap = expectedParser.map();

        String jsonRequest;
        try (XContentBuilder builder = JsonXContent.contentBuilder()) {
            request.toXContent(builder, ToXContentObject.EMPTY_PARAMS);
            jsonRequest = builder.toString();
        }

        XContentParser generatedParser = JsonXContent.jsonXContent.createParser(
            NamedXContentRegistry.EMPTY,
            DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
            jsonRequest
        );
        Map<String, Object> generatedMap = generatedParser.map();

        assertEquals(expectedMap, generatedMap);
    }

    /**
     * Get a mock JSON build request
     * <p>
     * Returns:
     * <pre>{@code {
     *   "repository_type": "s3",
     *   "container_name": "test-bucket",
     *   "vector_path": "vectors/1_1_25/SIRKos4rOWlMA62PX2p75m_vectors/SIRKos4rOWlMA62PX2p75m_target_field__3l.knnvec",
     *   "doc_id_path": "vectors/1_1_25/SIRKos4rOWlMA62PX2p75m_vectors/SIRKos4rOWlMA62PX2p75m_target_field__3l.knndid",
     *   "tenant_id": "test-cluster",
     *   "dimension": 2,
     *   "doc_count": 2,
     *   "data_type": "float",
     *   "engine": "faiss",
     *   "index_parameters": {
     *     "space_type": "l2",
     *     "algorithm": "hnsw",
     *     "algorithm_parameters": {
     *       "m": 14,
     *       "ef_construction": 94,
     *       "ef_search": 89
     *     }
     *   }
     * }}</pre>
     */
    public String getMockExpectedJson() {
        return "{"
            + "\""
            + REPOSITORY_TYPE
            + "\":\""
            + S3
            + "\","
            + "\""
            + CONTAINER_NAME
            + "\":\""
            + RemoteIndexHTTPClientTests.TEST_BUCKET
            + "\","
            + "\""
            + VECTOR_PATH
            + "\":\""
            + MOCK_FULL_PATH
            + VECTOR_BLOB_FILE_EXTENSION
            + "\","
            + "\""
            + DOC_ID_PATH
            + "\":\""
            + MOCK_FULL_PATH
            + DOC_ID_FILE_EXTENSION
            + "\","
            + "\""
            + TENANT_ID
            + "\":\""
            + RemoteIndexHTTPClientTests.TEST_CLUSTER
            + "\","
            + "\""
            + DIMENSION
            + "\":2,"
            + "\""
            + DOC_COUNT
            + "\":2,"
            + "\""
            + VECTOR_DATA_TYPE_FIELD
            + "\":\""
            + FLOAT
            + "\","
            + "\""
            + KNN_ENGINE
            + "\":\""
            + FAISS
            + "\","
            + "\""
            + INDEX_PARAMETERS
            + "\":{"
            + "\""
            + METHOD_PARAMETER_SPACE_TYPE
            + "\":\""
            + L2_SPACE_TYPE
            + "\","
            + "\""
            + ALGORITHM
            + "\":\""
            + HNSW_ALGORITHM
            + "\","
            + "\""
            + ALGORITHM_PARAMETERS
            + "\":{"
            + "\""
            + METHOD_PARAMETER_EF_CONSTRUCTION
            + "\":94,"
            + "\""
            + METHOD_PARAMETER_EF_SEARCH
            + "\":89,"
            + "\""
            + METHOD_PARAMETER_M
            + "\":14"
            + "}"
            + "}"
            + "}";
    }
}

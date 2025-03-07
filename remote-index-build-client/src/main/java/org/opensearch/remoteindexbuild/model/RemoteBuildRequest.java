/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.remoteindexbuild.model;

import lombok.Builder;
import lombok.Value;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;

import java.io.IOException;

/**
 * Request object for sending build requests to the remote build service, encapsulating all the required parameters
 * in a generic XContent format.
 */
@Value
@Builder
public class RemoteBuildRequest implements ToXContentObject {
    private static final String DOC_COUNT = "doc_count";
    private static final String TENANT_ID = "tenant_id";
    private static final String DOC_ID_PATH = "doc_id_path";
    private static final String DIMENSION = "dimension";
    private static final String VECTOR_DATA_TYPE_FIELD = "data_type";
    private static final String KNN_ENGINE = "engine";

    private static final String VECTOR_PATH = "vector_path";
    private static final String CONTAINER_NAME = "container_name";
    private static final String REPOSITORY_TYPE = "repository_type";
    private static final String INDEX_PARAMETERS = "index_parameters";

    String repositoryType;
    String containerName;
    String vectorPath;
    String docIdPath;
    String tenantId;
    int dimension;
    int docCount;
    String vectorDataType;
    String engine;
    RemoteIndexParameters indexParameters;

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field(REPOSITORY_TYPE, repositoryType);
        builder.field(CONTAINER_NAME, containerName);
        builder.field(VECTOR_PATH, vectorPath);
        builder.field(DOC_ID_PATH, docIdPath);
        builder.field(TENANT_ID, tenantId);
        builder.field(DIMENSION, dimension);
        builder.field(DOC_COUNT, docCount);
        builder.field(VECTOR_DATA_TYPE_FIELD, vectorDataType);
        builder.field(KNN_ENGINE, engine);
        builder.field(INDEX_PARAMETERS, indexParameters);
        builder.endObject();
        return builder;
    }
}

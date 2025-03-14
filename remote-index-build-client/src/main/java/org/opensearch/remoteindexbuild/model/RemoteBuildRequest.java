/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.remoteindexbuild.model;

import lombok.Builder;
import lombok.Getter;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;

import java.io.IOException;

import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.CONTAINER_NAME;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.DIMENSION;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.DOC_COUNT;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.DOC_ID_PATH;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.INDEX_PARAMETERS;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.KNN_ENGINE;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.REPOSITORY_TYPE;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.TENANT_ID;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.VECTOR_PATH;

/**
 * Request object for sending build requests to the remote build service, encapsulating all the required parameters
 * in a generic XContent format.
 */
@Getter
@Builder
public class RemoteBuildRequest implements ToXContentObject {
    protected String repositoryType;
    protected String containerName;
    protected String vectorPath;
    protected String docIdPath;
    protected String tenantId;
    protected int dimension;
    protected int docCount;
    protected String vectorDataType;
    protected String engine;
    protected RemoteIndexParameters indexParameters;

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

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import lombok.Getter;
import org.opensearch.cluster.metadata.RepositoryMetadata;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;

import java.io.IOException;

import static org.opensearch.knn.common.KNNConstants.CONTAINER_NAME;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.DOC_COUNT;
import static org.opensearch.knn.common.KNNConstants.DOC_ID_PATH;
import static org.opensearch.knn.common.KNNConstants.INDEX_PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.REPOSITORY_TYPE;
import static org.opensearch.knn.common.KNNConstants.TENANT_ID;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.common.KNNConstants.VECTOR_PATH;

/**
 * HTTP-specific implementation of RemoteBuildRequest.
 */
@Getter
public class HTTPRemoteBuildRequest extends RemoteBuildRequest {
    public HTTPRemoteBuildRequest(
        IndexSettings indexSettings,
        BuildIndexParams indexInfo,
        RepositoryMetadata repositoryMetadata,
        String blobName
    ) throws IOException {
        super(indexSettings, indexInfo, repositoryMetadata, blobName);
    }

    public String toJson() throws IOException {
        try (XContentBuilder builder = JsonXContent.contentBuilder()) {
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
            return builder.toString();
        }
    }
}

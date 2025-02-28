/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import lombok.Getter;
import org.opensearch.cluster.ClusterName;
import org.opensearch.cluster.metadata.RepositoryMetadata;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.codec.nativeindex.remote.RemoteIndexBuildStrategy;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.BUCKET;
import static org.opensearch.knn.common.KNNConstants.S3;

/**
 * Abstract base class for Remote Build Requests.
 */
@Getter
public abstract class RemoteBuildRequest {
    protected String repositoryType;
    protected String containerName;
    protected String vectorPath;
    protected String docIdPath;
    protected String tenantId;
    protected int dimension;
    protected int docCount;
    protected String vectorDataType;
    protected String engine;
    protected Map<String, Object> indexParameters;

    public RemoteBuildRequest(
        IndexSettings indexSettings,
        BuildIndexParams indexInfo,
        RepositoryMetadata repositoryMetadata,
        String blobName
    ) throws IOException {
        String repositoryType = repositoryMetadata.type();
        String containerName;
        switch (repositoryType) {
            case S3 -> containerName = repositoryMetadata.settings().get(BUCKET);
            default -> throw new IllegalArgumentException(
                "Repository type " + repositoryType + " is not supported by the remote build service"
            );
        }
        String vectorDataType = indexInfo.getVectorDataType().getValue();

        KNNVectorValues<?> vectorValues = indexInfo.getKnnVectorValuesSupplier().get();
        KNNCodecUtil.initializeVectorValues(vectorValues);
        assert (vectorValues.dimension() > 0);

        Map<String, Object> indexParameters = indexInfo.getKnnEngine().getRemoteIndexingParameters(indexInfo.getParameters());

        this.repositoryType = repositoryType;
        this.containerName = containerName;
        this.vectorPath = blobName + RemoteIndexBuildStrategy.VECTOR_BLOB_FILE_EXTENSION;
        this.docIdPath = blobName + RemoteIndexBuildStrategy.DOC_ID_FILE_EXTENSION;
        this.tenantId = indexSettings.getSettings().get(ClusterName.CLUSTER_NAME_SETTING.getKey());
        this.dimension = vectorValues.dimension();
        this.docCount = indexInfo.getTotalLiveDocs();
        this.vectorDataType = vectorDataType;
        this.engine = indexInfo.getKnnEngine().getName();
        this.indexParameters = indexParameters;
    }

    /**
     * Convert the request to JSON format.
     */
    public abstract String toJson() throws IOException;
}

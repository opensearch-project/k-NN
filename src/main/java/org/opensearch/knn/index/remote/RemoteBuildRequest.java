/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import lombok.Getter;
import org.opensearch.cluster.ClusterName;
import org.opensearch.cluster.metadata.RepositoryMetadata;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.BUCKET;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.DOC_ID_FILE_EXTENSION;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.INDEX_PARAMETERS;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.S3;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.VECTOR_BLOB_FILE_EXTENSION;

/**
 * Request object for sending build requests to the remote build service, encapsulating all the required parameters
 * in a generic XContent format.
 */
@Getter
public class RemoteBuildRequest implements ToXContentObject {
    private final String DOC_COUNT = "doc_count";
    private final String TENANT_ID = "tenant_id";
    private final String DOC_ID_PATH = "doc_id_path";
    public final String VECTOR_PATH = "vector_path";
    public final String CONTAINER_NAME = "container_name";
    public final String REPOSITORY_TYPE = "repository_type";

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

    /**
     * Constructor for RemoteBuildRequest.
     *
     * @param indexSettings IndexSettings object
     * @param indexInfo BuildIndexParams object
     * @param repositoryMetadata RepositoryMetadata object
     * @param blobName Name of the blob
     * @throws IOException if an I/O error occurs
     */
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

        this.repositoryType = repositoryType;
        this.containerName = containerName;
        this.vectorPath = blobName + VECTOR_BLOB_FILE_EXTENSION;
        this.docIdPath = blobName + DOC_ID_FILE_EXTENSION;
        this.tenantId = indexSettings.getSettings().get(ClusterName.CLUSTER_NAME_SETTING.getKey());
        this.dimension = vectorValues.dimension();
        this.docCount = indexInfo.getTotalLiveDocs();
        this.vectorDataType = vectorDataType;
        this.engine = indexInfo.getKnnEngine().getName();
        this.indexParameters = indexInfo.getKnnEngine().createRemoteIndexingParameters(indexInfo.getParameters());
    }

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

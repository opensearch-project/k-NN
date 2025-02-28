/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.remote;

import org.opensearch.cluster.metadata.RepositoryMetadata;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;

import java.io.IOException;

/**
 * Generic Builder for constructing RemoteBuildRequest objects per requestType (HTTP, etc).
 */
public class RemoteBuildRequestBuilder<T extends RemoteBuildRequest> {
    private IndexSettings indexSettings;
    private BuildIndexParams indexInfo;
    private RepositoryMetadata repositoryMetadata;
    private String blobName;
    private final Class<T> requestType;

    // Private constructor
    private RemoteBuildRequestBuilder(Class<T> requestType) {
        this.requestType = requestType;
    }

    /**
     * Static factory method to create a builder instance.
     * @param requestType Class type of the RemoteBuildRequest
     * @return RemoteBuildRequestBuilder instance
     */
    public static <T extends RemoteBuildRequest> RemoteBuildRequestBuilder<T> builder(Class<T> requestType) {
        return new RemoteBuildRequestBuilder<>(requestType);
    }

    public RemoteBuildRequestBuilder<T> indexSettings(IndexSettings indexSettings) {
        this.indexSettings = indexSettings;
        return this;
    }

    public RemoteBuildRequestBuilder<T> indexInfo(BuildIndexParams indexInfo) {
        this.indexInfo = indexInfo;
        return this;
    }

    public RemoteBuildRequestBuilder<T> repositoryMetadata(RepositoryMetadata repositoryMetadata) {
        this.repositoryMetadata = repositoryMetadata;
        return this;
    }

    public RemoteBuildRequestBuilder<T> blobName(String blobName) {
        this.blobName = blobName;
        return this;
    }

    public T build() throws IOException {
        try {
            return requestType.getConstructor(IndexSettings.class, BuildIndexParams.class, RepositoryMetadata.class, String.class)
                .newInstance(indexSettings, indexInfo, repositoryMetadata, blobName);
        } catch (Exception e) {
            throw new IOException("Failed to instantiate RemoteBuildRequest", e);
        }
    }
}

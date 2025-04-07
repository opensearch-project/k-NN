/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import lombok.extern.log4j.Log4j2;
import org.opensearch.cluster.ClusterName;
import org.opensearch.cluster.metadata.RepositoryMetadata;
import org.opensearch.common.UUIDs;
import org.opensearch.common.annotation.ExperimentalApi;
import org.opensearch.common.blobstore.BlobPath;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategy;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.remote.RemoteIndexWaiter;
import org.opensearch.knn.index.remote.RemoteIndexWaiterFactory;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.remoteindexbuild.client.RemoteIndexClient;
import org.opensearch.remoteindexbuild.client.RemoteIndexClientFactory;
import org.opensearch.remoteindexbuild.model.RemoteBuildRequest;
import org.opensearch.remoteindexbuild.model.RemoteBuildResponse;
import org.opensearch.remoteindexbuild.model.RemoteBuildStatusRequest;
import org.opensearch.remoteindexbuild.model.RemoteBuildStatusResponse;
import org.opensearch.repositories.RepositoriesService;
import org.opensearch.repositories.Repository;
import org.opensearch.repositories.RepositoryMissingException;
import org.opensearch.repositories.blobstore.BlobStoreRepository;

import java.io.IOException;
import java.util.Map;
import java.util.function.Supplier;

import static org.opensearch.knn.common.KNNConstants.BUCKET;
import static org.opensearch.knn.common.KNNConstants.DOC_ID_FILE_EXTENSION;
import static org.opensearch.knn.common.KNNConstants.S3;
import static org.opensearch.knn.common.KNNConstants.VECTORS_PATH;
import static org.opensearch.knn.common.KNNConstants.VECTOR_BLOB_FILE_EXTENSION;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX_REMOTE_VECTOR_BUILD_SETTING;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX_REMOTE_VECTOR_BUILD_THRESHOLD_SETTING;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_VECTOR_REPO_SETTING;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;

/**
 * This class orchestrates building vector indices. It handles uploading data to a repository, submitting a remote
 * build request, awaiting upon the build request to complete, and finally downloading the data from a repository.
 */
@Log4j2
@ExperimentalApi
public class RemoteIndexBuildStrategy implements NativeIndexBuildStrategy {

    private final Supplier<RepositoriesService> repositoriesServiceSupplier;
    private final NativeIndexBuildStrategy fallbackStrategy;
    private final IndexSettings indexSettings;
    private final KNNLibraryIndexingContext knnLibraryIndexingContext;
    private final RemoteIndexBuildMetrics metrics;

    /**
     * Public constructor, intended to be called by {@link org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory} based in
     * part on the return value from {@link RemoteIndexBuildStrategy#shouldBuildIndexRemotely}
     *
     * @param repositoriesServiceSupplier A supplier for {@link RepositoriesService} used to interact with a repository
     * @param fallbackStrategy            Delegate {@link NativeIndexBuildStrategy} used to fall back to local build
     * @param indexSettings               {@link IndexSettings} used to retrieve information about the index
     * @param knnLibraryIndexingContext   {@link KNNLibraryIndexingContext} used to retrieve method specific params for the remote build request
     */
    public RemoteIndexBuildStrategy(
        Supplier<RepositoriesService> repositoriesServiceSupplier,
        NativeIndexBuildStrategy fallbackStrategy,
        IndexSettings indexSettings,
        KNNLibraryIndexingContext knnLibraryIndexingContext
    ) {
        this.repositoriesServiceSupplier = repositoriesServiceSupplier;
        this.fallbackStrategy = fallbackStrategy;
        this.indexSettings = indexSettings;
        this.knnLibraryIndexingContext = knnLibraryIndexingContext;
        this.metrics = new RemoteIndexBuildMetrics();
    }

    /**
     * @param indexSettings         {@link IndexSettings} used to check if index setting is enabled for the feature
     * @param vectorBlobLength      The size of the vector blob, used to determine if the size threshold is met
     * @return true if remote index build should be used, else false
     */
    public static boolean shouldBuildIndexRemotely(IndexSettings indexSettings, long vectorBlobLength) {
        if (indexSettings == null) {
            return false;
        }

        // If setting is not enabled, return false
        if (!indexSettings.getValue(KNN_INDEX_REMOTE_VECTOR_BUILD_SETTING)) {
            log.debug("Remote index build is disabled for index: [{}]", indexSettings.getIndex().getName());
            return false;
        }

        // If vector repo is not configured, return false
        String vectorRepo = KNNSettings.state().getSettingValue(KNN_REMOTE_VECTOR_REPO_SETTING.getKey());
        if (vectorRepo == null || vectorRepo.isEmpty()) {
            log.debug("Vector repo is not configured, falling back to local build for index: [{}]", indexSettings.getIndex().getName());
            return false;
        }

        // If size threshold is not met, return false
        if (vectorBlobLength < indexSettings.getValue(KNN_INDEX_REMOTE_VECTOR_BUILD_THRESHOLD_SETTING).getBytes()) {
            log.debug(
                "Data size [{}] is less than remote index build threshold [{}], falling back to local build for index [{}]",
                vectorBlobLength,
                indexSettings.getValue(KNN_INDEX_REMOTE_VECTOR_BUILD_THRESHOLD_SETTING).getBytes(),
                indexSettings.getIndex().getName()
            );
            return false;
        }

        return true;
    }

    /**
     * Entry point for flush/merge operations. This method orchestrates the following:
     *      1. Writes required data to repository
     *      2. Triggers index build
     *      3. Awaits on vector build to complete
     *      4. Downloads index file and writes to indexOutput
     *
     * @param indexInfo {@link BuildIndexParams} containing information about the index to be built
     * @throws IOException if an error occurs during the build process
     */
    @Override
    public void buildAndWriteIndex(BuildIndexParams indexInfo) throws IOException {
        metrics.startRemoteIndexBuildMetrics(indexInfo);
        boolean success = false;
        try {
            RepositoryContext repositoryContext = getRepositoryContext(indexInfo);

            // 1. Write required data to repository
            writeToRepository(repositoryContext, indexInfo);

            // 2. Trigger remote index build
            RemoteIndexClient client = RemoteIndexClientFactory.getRemoteIndexClient(KNNSettings.getRemoteBuildServiceEndpoint());
            RemoteBuildResponse remoteBuildResponse = submitBuild(repositoryContext, indexInfo, client);

            // 3. Await vector build completion
            RemoteBuildStatusResponse remoteBuildStatusResponse = awaitIndexBuild(remoteBuildResponse, indexInfo, client);

            // 4. Download index file and write to indexOutput
            readFromRepository(indexInfo, repositoryContext, remoteBuildStatusResponse);

            success = true;
        } catch (Exception e) {
            fallbackStrategy.buildAndWriteIndex(indexInfo);
        } finally {
            metrics.endRemoteIndexBuildMetrics(success);
        }
    }

    /**
     * Writes the required vector and doc ID data to the repository
     */
    private void writeToRepository(RepositoryContext repositoryContext, BuildIndexParams indexInfo) throws IOException,
        InterruptedException {
        VectorRepositoryAccessor vectorRepositoryAccessor = repositoryContext.vectorRepositoryAccessor;
        boolean success = false;
        metrics.startRepositoryWriteMetrics();
        try {
            vectorRepositoryAccessor.writeToRepository(
                repositoryContext.blobName,
                indexInfo.getTotalLiveDocs(),
                indexInfo.getVectorDataType(),
                indexInfo.getKnnVectorValuesSupplier()
            );
            success = true;
        } catch (InterruptedException | IOException e) {
            log.debug("Repository write failed for vector field [{}]", indexInfo.getFieldName());
            throw e;
        } finally {
            metrics.endRepositoryWriteMetrics(success);
        }
    }

    /**
     * Submits a remote build request to the remote index build service
     * @return RemoteBuildResponse containing the response from the remote service
     */
    private RemoteBuildResponse submitBuild(RepositoryContext repositoryContext, BuildIndexParams indexInfo, RemoteIndexClient client)
        throws IOException {
        final RemoteBuildResponse remoteBuildResponse;
        boolean success = false;
        metrics.startBuildRequestMetrics();
        try {
            final RemoteBuildRequest buildRequest = buildRemoteBuildRequest(
                indexSettings,
                indexInfo,
                repositoryContext.blobStoreRepository.getMetadata(),
                repositoryContext.blobPath.buildAsString() + repositoryContext.blobName,
                knnLibraryIndexingContext.getLibraryParameters()
            );
            remoteBuildResponse = client.submitVectorBuild(buildRequest);
            success = true;
            return remoteBuildResponse;
        } catch (IOException e) {
            log.debug("Submit vector build failed for vector field [{}]", indexInfo.getFieldName());
            throw e;
        } finally {
            metrics.endBuildRequestMetrics(success);
        }
    }

    /**
     * Awaits the vector build to complete
     * @return RemoteBuildStatusResponse containing the completed status response from the remote service.
     * This will only be returned with a COMPLETED_INDEX_BUILD status, otherwise the method will throw an exception.
     */
    private RemoteBuildStatusResponse awaitIndexBuild(
        RemoteBuildResponse remoteBuildResponse,
        BuildIndexParams indexInfo,
        RemoteIndexClient client
    ) throws IOException, InterruptedException {
        RemoteBuildStatusResponse remoteBuildStatusResponse;
        metrics.startWaitingMetrics();
        try {
            final RemoteBuildStatusRequest remoteBuildStatusRequest = RemoteBuildStatusRequest.builder()
                .jobId(remoteBuildResponse.getJobId())
                .build();
            RemoteIndexWaiter waiter = RemoteIndexWaiterFactory.getRemoteIndexWaiter(client);
            remoteBuildStatusResponse = waiter.awaitVectorBuild(remoteBuildStatusRequest);
            metrics.endWaitingMetrics();
            return remoteBuildStatusResponse;
        } catch (InterruptedException | IOException e) {
            log.debug("Await vector build failed for vector field [{}]", indexInfo.getFieldName(), e);
            throw e;
        }
    }

    /**
     * Downloads the index file from the repository and writes to the indexOutput
     */
    private void readFromRepository(
        BuildIndexParams indexInfo,
        RepositoryContext repositoryContext,
        RemoteBuildStatusResponse remoteBuildStatusResponse
    ) throws IOException {
        metrics.startRepositoryReadMetrics();
        boolean success = false;
        try {
            repositoryContext.vectorRepositoryAccessor.readFromRepository(
                remoteBuildStatusResponse.getFileName(),
                indexInfo.getIndexOutputWithBuffer()
            );
            success = true;
        } catch (Exception e) {
            log.debug("Repository read failed for vector field [{}]", indexInfo.getFieldName());
            throw e;
        } finally {
            metrics.endRepositoryReadMetrics(success);
        }
    }

    /**
     * @return {@link BlobStoreRepository} referencing the repository
     * @throws RepositoryMissingException if repository is not registered or if {@link KNNSettings#KNN_REMOTE_VECTOR_REPO_SETTING} is not set
     */
    private BlobStoreRepository getRepository() throws RepositoryMissingException {
        RepositoriesService repositoriesService = repositoriesServiceSupplier.get();
        assert repositoriesService != null;
        String vectorRepo = KNNSettings.state().getSettingValue(KNN_REMOTE_VECTOR_REPO_SETTING.getKey());
        if (vectorRepo == null || vectorRepo.isEmpty()) {
            throw new RepositoryMissingException("Vector repository " + KNN_REMOTE_VECTOR_REPO_SETTING.getKey() + " is not registered");
        }
        final Repository repository = repositoriesService.repository(vectorRepo);
        assert repository instanceof BlobStoreRepository : "Repository should be instance of BlobStoreRepository";
        return (BlobStoreRepository) repository;
    }

    /**
     * Record to hold various repository related objects
     */
    private record RepositoryContext(BlobStoreRepository blobStoreRepository, BlobPath blobPath,
        VectorRepositoryAccessor vectorRepositoryAccessor, String blobName) {
    }

    /**
     * Helper method to get repository context. Generates a unique UUID for the blobName so should only be used once.
     */
    private RepositoryContext getRepositoryContext(BuildIndexParams indexInfo) {
        BlobStoreRepository repository = getRepository();
        BlobPath blobPath = repository.basePath().add(indexSettings.getUUID() + VECTORS_PATH);
        String blobName = UUIDs.base64UUID() + "_" + indexInfo.getFieldName() + "_" + indexInfo.getSegmentWriteState().segmentInfo.name;
        VectorRepositoryAccessor vectorRepositoryAccessor = new DefaultVectorRepositoryAccessor(
            repository.blobStore().blobContainer(blobPath)
        );
        return new RepositoryContext(repository, blobPath, vectorRepositoryAccessor, blobName);
    }

    /**
     * Constructor for RemoteBuildRequest.
     *
     * @param indexSettings      IndexSettings object
     * @param indexInfo          BuildIndexParams object
     * @param repositoryMetadata RepositoryMetadata object
     * @param fullPath           Full blob path + file name representing location of the vectors/doc IDs (excludes repository-specific prefix)
     * @param parameters         Map of parameters to be parsed and passed to the remote build service
     * @throws IOException if an I/O error occurs
     */
    static RemoteBuildRequest buildRemoteBuildRequest(
        IndexSettings indexSettings,
        BuildIndexParams indexInfo,
        RepositoryMetadata repositoryMetadata,
        String fullPath,
        Map<String, Object> parameters
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
        initializeVectorValues(vectorValues);
        assert (vectorValues.dimension() > 0);

        return RemoteBuildRequest.builder()
            .repositoryType(repositoryType)
            .containerName(containerName)
            .vectorPath(fullPath + VECTOR_BLOB_FILE_EXTENSION)
            .docIdPath(fullPath + DOC_ID_FILE_EXTENSION)
            .tenantId(indexSettings.getSettings().get(ClusterName.CLUSTER_NAME_SETTING.getKey()))
            .dimension(vectorValues.dimension())
            .docCount(indexInfo.getTotalLiveDocs())
            .vectorDataType(vectorDataType)
            .engine(indexInfo.getKnnEngine().getName())
            .indexParameters(indexInfo.getKnnEngine().createRemoteIndexingParameters(parameters))
            .build();
    }
}

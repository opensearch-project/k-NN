/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import lombok.extern.log4j.Log4j2;
import org.opensearch.cluster.ClusterName;
import org.opensearch.cluster.metadata.RepositoryMetadata;
import org.opensearch.common.StopWatch;
import org.opensearch.common.UUIDs;
import org.opensearch.common.annotation.ExperimentalApi;
import org.opensearch.common.blobstore.BlobPath;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategy;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
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
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.BUILD_REQUEST_FAILURE_COUNT;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.BUILD_REQUEST_SUCCESS_COUNT;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.INDEX_BUILD_FAILURE_COUNT;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.INDEX_BUILD_SUCCESS_COUNT;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.READ_FAILURE_COUNT;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.READ_SUCCESS_COUNT;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.READ_TIME;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_CURRENT_OPERATIONS;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_CURRENT_SIZE;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_TIME;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.WAITING_TIME;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.WRITE_FAILURE_COUNT;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.WRITE_SUCCESS_COUNT;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.WRITE_TIME;

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

    /**
     * Public constructor, intended to be called by {@link org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory} based in
     * part on the return value from {@link RemoteIndexBuildStrategy#shouldBuildIndexRemotely}
     * @param repositoriesServiceSupplier       A supplier for {@link RepositoriesService} used to interact with a repository
     * @param fallbackStrategy                  Delegate {@link NativeIndexBuildStrategy} used to fall back to local build
     * @param indexSettings                    {@link IndexSettings} used to retrieve information about the index
     * @param knnLibraryIndexingContext                 {@link KNNLibraryIndexingContext} used to retrieve method specific params for the remote build request
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
     */
    @Override
    public void buildAndWriteIndex(BuildIndexParams indexInfo) throws IOException {
        StopWatch remoteBuildStopWatch = new StopWatch();
        KNNVectorValues<?> knnVectorValues = indexInfo.getKnnVectorValuesSupplier().get();
        initializeVectorValues(knnVectorValues);
        startRemoteIndexBuildStats((long) indexInfo.getTotalLiveDocs() * knnVectorValues.bytesPerVector(), remoteBuildStopWatch);

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

            endRemoteIndexBuildStats((long) indexInfo.getTotalLiveDocs() * knnVectorValues.bytesPerVector(), remoteBuildStopWatch);
            INDEX_BUILD_SUCCESS_COUNT.increment();
        } catch (Exception e) {
            log.warn("Failed to build index remotely", e);
            handleFailure(indexInfo, knnVectorValues.bytesPerVector(), remoteBuildStopWatch);
        }
    }

    /**
     * Writes the required vector and doc ID data to the repository
     */
    private void writeToRepository(RepositoryContext repositoryContext, BuildIndexParams indexInfo) throws IOException,
        InterruptedException {
        BlobStoreRepository repository = repositoryContext.blobStoreRepository;
        BlobPath blobPath = repositoryContext.blobPath;
        VectorRepositoryAccessor vectorRepositoryAccessor = new DefaultVectorRepositoryAccessor(
            repository.blobStore().blobContainer(blobPath)
        );

        StopWatch stopWatch = new StopWatch().start();
        long time_in_millis;
        try {
            vectorRepositoryAccessor.writeToRepository(
                repositoryContext.blobName,
                indexInfo.getTotalLiveDocs(),
                indexInfo.getVectorDataType(),
                indexInfo.getKnnVectorValuesSupplier()
            );
            time_in_millis = stopWatch.stop().totalTime().millis();
            WRITE_SUCCESS_COUNT.increment();
            WRITE_TIME.incrementBy(time_in_millis);
            log.debug("Repository write took {} ms for vector field [{}]", time_in_millis, indexInfo.getFieldName());
        } catch (Exception e) {
            time_in_millis = stopWatch.stop().totalTime().millis();
            WRITE_FAILURE_COUNT.increment();
            log.error("Repository write failed after {} ms for vector field [{}]", time_in_millis, indexInfo.getFieldName(), e);
            throw e;
        }
    }

    /**
     * Submits a remote build request to the remote index build service
     * @return RemoteBuildResponse containing the response from the remote service
     */
    private RemoteBuildResponse submitBuild(RepositoryContext repositoryContext, BuildIndexParams indexInfo, RemoteIndexClient client)
        throws IOException {
        final RemoteBuildResponse remoteBuildResponse;
        StopWatch stopWatch = new StopWatch().start();
        long time_in_millis;
        try {
            final RemoteBuildRequest buildRequest = buildRemoteBuildRequest(
                indexSettings,
                indexInfo,
                repositoryContext.blobStoreRepository.getMetadata(),
                repositoryContext.blobPath.buildAsString() + repositoryContext.blobName,
                knnLibraryIndexingContext.getLibraryParameters()
            );
            remoteBuildResponse = client.submitVectorBuild(buildRequest);
            BUILD_REQUEST_SUCCESS_COUNT.increment();
            time_in_millis = stopWatch.stop().totalTime().millis();
            log.debug("Submit vector build took {} ms for vector field [{}]", time_in_millis, indexInfo.getFieldName());
            return remoteBuildResponse;
        } catch (IOException e) {
            BUILD_REQUEST_FAILURE_COUNT.increment();
            time_in_millis = stopWatch.stop().totalTime().millis();
            log.error("Submit vector build failed after {} ms for vector field [{}]", time_in_millis, indexInfo.getFieldName(), e);
            throw e;
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
        StopWatch stopWatch = new StopWatch().start();
        long time_in_millis;
        try {
            final RemoteBuildStatusRequest remoteBuildStatusRequest = RemoteBuildStatusRequest.builder()
                .jobId(remoteBuildResponse.getJobId())
                .build();
            RemoteIndexWaiter waiter = RemoteIndexWaiterFactory.getRemoteIndexWaiter(client);
            remoteBuildStatusResponse = waiter.awaitVectorBuild(remoteBuildStatusRequest);
            time_in_millis = stopWatch.stop().totalTime().millis();
            WAITING_TIME.incrementBy(time_in_millis);
            log.debug("Await vector build took {} ms for vector field [{}]", time_in_millis, indexInfo.getFieldName());
            return remoteBuildStatusResponse;
        } catch (InterruptedException | IOException e) {
            time_in_millis = stopWatch.stop().totalTime().millis();
            log.error("Await vector build failed after {} ms for vector field [{}]", time_in_millis, indexInfo.getFieldName(), e);
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
        StopWatch stopWatch = new StopWatch().start();
        long time_in_millis;
        try {
            repositoryContext.vectorRepositoryAccessor.readFromRepository(
                remoteBuildStatusResponse.getFileName(),
                indexInfo.getIndexOutputWithBuffer()
            );
            time_in_millis = stopWatch.stop().totalTime().millis();
            READ_SUCCESS_COUNT.increment();
            READ_TIME.incrementBy(time_in_millis);
            log.debug("Repository read took {} ms for vector field [{}]", time_in_millis, indexInfo.getFieldName());
        } catch (Exception e) {
            time_in_millis = stopWatch.stop().totalTime().millis();
            READ_FAILURE_COUNT.increment();
            log.error("Repository read failed after {} ms for vector field [{}]", time_in_millis, indexInfo.getFieldName(), e);
            throw e;
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

    /**
     * Helper method to collect remote index build metrics on start
     */
    private void startRemoteIndexBuildStats(long size, StopWatch stopWatch) {
        stopWatch.start();
        REMOTE_INDEX_BUILD_CURRENT_OPERATIONS.increment();
        REMOTE_INDEX_BUILD_CURRENT_SIZE.incrementBy(size);
    }

    /**
     * Helper method to collect remote index build metrics on success
     */
    private void endRemoteIndexBuildStats(long size, StopWatch stopWatch) {
        long time_in_millis = stopWatch.stop().totalTime().millis();
        REMOTE_INDEX_BUILD_CURRENT_OPERATIONS.decrement();
        REMOTE_INDEX_BUILD_CURRENT_SIZE.decrementBy(size);
        REMOTE_INDEX_BUILD_TIME.incrementBy(time_in_millis);
    }

    /**
     * Helper method to collect remote index build metrics on failure and invoke fallback strategy
     */
    private void handleFailure(BuildIndexParams indexParams, long bytesPerVector, StopWatch stopWatch) throws IOException {
        endRemoteIndexBuildStats(indexParams.getTotalLiveDocs() * bytesPerVector, stopWatch);
        INDEX_BUILD_FAILURE_COUNT.increment();
        fallbackStrategy.buildAndWriteIndex(indexParams);
    }

    /**
     * Record to hold various repository related objects
     */
    private record RepositoryContext(BlobStoreRepository blobStoreRepository, BlobPath blobPath,
        VectorRepositoryAccessor vectorRepositoryAccessor, String blobName) {
    }

    /**
     * Helper method to get repository context
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
}

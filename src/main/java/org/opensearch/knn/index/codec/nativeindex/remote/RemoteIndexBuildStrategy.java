/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import lombok.extern.log4j.Log4j2;
import org.opensearch.common.StopWatch;
import org.opensearch.common.UUIDs;
import org.opensearch.common.annotation.ExperimentalApi;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategy;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.remote.RemoteBuildRequest;
import org.opensearch.knn.index.remote.RemoteBuildResponse;
import org.opensearch.knn.index.remote.RemoteIndexClient;
import org.opensearch.knn.index.remote.RemoteIndexClientFactory;
import org.opensearch.knn.index.remote.RemoteStatusResponse;
import org.opensearch.repositories.RepositoriesService;
import org.opensearch.repositories.Repository;
import org.opensearch.repositories.RepositoryMissingException;
import org.opensearch.repositories.blobstore.BlobStoreRepository;

import java.io.IOException;
import java.util.function.Supplier;

import static org.opensearch.knn.index.KNNSettings.KNN_INDEX_REMOTE_VECTOR_BUILD_SETTING;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX_REMOTE_VECTOR_BUILD_THRESHOLD_SETTING;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_VECTOR_REPO_SETTING;

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

    /**
     * Public constructor, intended to be called by {@link org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory} based in
     * part on the return value from {@link RemoteIndexBuildStrategy#shouldBuildIndexRemotely}
     * @param repositoriesServiceSupplier       A supplier for {@link RepositoriesService} used to interact with a repository
     * @param fallbackStrategy                  Delegate {@link NativeIndexBuildStrategy} used to fall back to local build
     * @param indexSettings                    {@link IndexSettings} used to retrieve information about the index
     */
    public RemoteIndexBuildStrategy(
        Supplier<RepositoriesService> repositoriesServiceSupplier,
        NativeIndexBuildStrategy fallbackStrategy,
        IndexSettings indexSettings
    ) {
        this.repositoriesServiceSupplier = repositoriesServiceSupplier;
        this.fallbackStrategy = fallbackStrategy;
        this.indexSettings = indexSettings;
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
     * @param indexInfo
     * @throws IOException
     */
    @Override
    public void buildAndWriteIndex(BuildIndexParams indexInfo) throws IOException {
        StopWatch stopWatch;
        long time_in_millis;
        try {
            VectorRepositoryAccessor vectorRepositoryAccessor = new DefaultVectorRepositoryAccessor(getRepository(), indexSettings);
            stopWatch = new StopWatch().start();
            // We create a new time based UUID per file in order to avoid conflicts across shards. It is also very difficult to get the
            // shard id in this context.
            String blobName = UUIDs.base64UUID() + "_" + indexInfo.getFieldName() + "_" + indexInfo.getSegmentWriteState().segmentInfo.name;
            vectorRepositoryAccessor.writeToRepository(
                blobName,
                indexInfo.getTotalLiveDocs(),
                indexInfo.getVectorDataType(),
                indexInfo.getKnnVectorValuesSupplier()
            );
            time_in_millis = stopWatch.stop().totalTime().millis();
            log.debug("Repository write took {} ms for vector field [{}]", time_in_millis, indexInfo.getFieldName());

            RemoteIndexClient client = RemoteIndexClientFactory.getRemoteIndexClient();
            RemoteBuildRequest request = new RemoteBuildRequest(indexSettings, indexInfo, getRepository().getMetadata(), blobName);
            stopWatch = new StopWatch().start();
            RemoteBuildResponse remoteBuildResponse = client.submitVectorBuild(request);
            time_in_millis = stopWatch.stop().totalTime().millis();
            log.debug("Submit vector build took {} ms for vector field [{}]", time_in_millis, indexInfo.getFieldName());

            stopWatch = new StopWatch().start();
            RemoteStatusResponse remoteStatusResponse = client.awaitVectorBuild(remoteBuildResponse);
            time_in_millis = stopWatch.stop().totalTime().millis();
            log.debug("Await vector build took {} ms for vector field [{}]", time_in_millis, indexInfo.getFieldName());

            stopWatch = new StopWatch().start();
            vectorRepositoryAccessor.readFromRepository(remoteStatusResponse.getIndexPath(), indexInfo.getIndexOutputWithBuffer());
            time_in_millis = stopWatch.stop().totalTime().millis();
            log.debug("Repository read took {} ms for vector field [{}]", time_in_millis, indexInfo.getFieldName());
        } catch (Exception e) {
            // TODO: This needs more robust failure handling
            log.warn("Failed to build index remotely", e);
            fallbackStrategy.buildAndWriteIndex(indexInfo);
        }
    }

    /**
     * Gets the KNN repository container from the repository service.
     *
     * @return {@link RepositoriesService}
     * @throws RepositoryMissingException if repository is not registered or if {@link KNN_REMOTE_VECTOR_REPO_SETTING} is not set
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
}

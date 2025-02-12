/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import lombok.extern.log4j.Log4j2;
import org.apache.commons.lang.NotImplementedException;
import org.apache.lucene.index.SegmentWriteState;
import org.opensearch.common.StopWatch;
import org.opensearch.common.annotation.ExperimentalApi;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.common.featureflags.KNNFeatureFlags;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategy;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.remote.RemoteBuildRequest;
import org.opensearch.knn.index.remote.RemoteIndexClient;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.repositories.RepositoriesService;
import org.opensearch.repositories.Repository;
import org.opensearch.repositories.RepositoryMissingException;
import org.opensearch.repositories.blobstore.BlobStoreRepository;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;

import static org.opensearch.knn.index.KNNSettings.KNN_INDEX_REMOTE_VECTOR_BUILD_SETTING;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_VECTOR_REPO_SETTING;
import static org.opensearch.knn.index.KNNSettings.state;

/**
 * This class orchestrates building vector indices. It handles uploading data to a repository, submitting a remote
 * build request, awaiting upon the build request to complete, and finally downloading the data from a repository.
 */
@Log4j2
@ExperimentalApi
public class RemoteIndexBuildStrategy implements NativeIndexBuildStrategy {

    private final Supplier<RepositoriesService> repositoriesServiceSupplier;
    private final NativeIndexBuildStrategy fallbackStrategy;
    private static final String VECTOR_BLOB_FILE_EXTENSION = ".knnvec";
    private static final String DOC_ID_FILE_EXTENSION = ".knndid";

    /**
     * Public constructor
     *
     * @param repositoriesServiceSupplier   A supplier for {@link RepositoriesService} used for interacting with repository
     */
    public RemoteIndexBuildStrategy(Supplier<RepositoriesService> repositoriesServiceSupplier, NativeIndexBuildStrategy fallbackStrategy) {
        this.repositoriesServiceSupplier = repositoriesServiceSupplier;
        this.fallbackStrategy = fallbackStrategy;
    }

    /**
     * @return whether to use the remote build feature
     */
    public static boolean shouldBuildIndexRemotely(IndexSettings indexSettings) {
        String vectorRepo = state().getSettingValue(KNN_REMOTE_VECTOR_REPO_SETTING.getKey());
        return KNNFeatureFlags.isKNNRemoteVectorBuildEnabled()
            && indexSettings.getValue(KNN_INDEX_REMOTE_VECTOR_BUILD_SETTING)
            && vectorRepo != null
            && !vectorRepo.isEmpty();
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
        // TODO: Metrics Collection
        StopWatch stopWatch;
        long time_in_millis;
        try {
            stopWatch = new StopWatch().start();
            writeToRepository(
                indexInfo.getFieldName(),
                indexInfo.getKnnVectorValuesSupplier(),
                indexInfo.getTotalLiveDocs(),
                indexInfo.getSegmentWriteState()
            );
            time_in_millis = stopWatch.stop().totalTime().millis();
            log.debug("Repository write took {} ms for vector field [{}]", time_in_millis, indexInfo.getFieldName());

            stopWatch = new StopWatch().start();
            RemoteBuildRequest buildRequest = constructBuildRequest(indexInfo);
            String jobId = RemoteIndexClient.getInstance().submitVectorBuild(buildRequest);
            time_in_millis = stopWatch.stop().totalTime().millis();
            log.debug("Submit vector build took {} ms for vector field [{}]", time_in_millis, indexInfo.getFieldName());

            stopWatch = new StopWatch().start();
            String indexPath = awaitVectorBuild(jobId);
            time_in_millis = stopWatch.stop().totalTime().millis();
            log.debug("Await vector build took {} ms for vector field [{}]", time_in_millis, indexInfo.getFieldName());

            stopWatch = new StopWatch().start();
            readFromRepository(indexPath);
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

    /**
     * Write relevant vector data to repository
     *
     * @param fieldName
     * @param knnVectorValuesSupplier
     * @param totalLiveDocs
     * @param segmentWriteState
     * @throws IOException
     * @throws InterruptedException
     */
    private void writeToRepository(
        String fieldName,
        Supplier<KNNVectorValues<?>> knnVectorValuesSupplier,
        int totalLiveDocs,
        SegmentWriteState segmentWriteState
    ) throws IOException, InterruptedException {
        throw new NotImplementedException();
    }

    /**
     * Construct the RemoteBuildRequest object for the index build request
     * @return RemoteBuildRequest with parameters set
     */
    public RemoteBuildRequest constructBuildRequest(BuildIndexParams indexInfo) throws IOException {
        String repositoryType = getRepository().getMetadata().type();
        String containerName = switch (repositoryType) {
            case "s3" -> getRepository().getMetadata().settings().get("bucket");
            case "fs" -> getRepository().getMetadata().settings().get("location");
            default -> throw new IllegalStateException("Unexpected value: " + repositoryType);
        };
        String vectorPath = null; // blobName + VECTOR_BLOB_FILE_EXTENSION
        String docIdPath = null; // blobName + DOC_ID_FILE_EXTENSION
        String tenantId = null; // indexSettings.getSettings().get(ClusterName.CLUSTER_NAME_SETTING.getKey());
        int dimension = 0; // TODO
        int docCount = indexInfo.getTotalLiveDocs();
        String dataType = indexInfo.getVectorDataType().getValue(); // TODO need to fetch encoder param to get fp16 vs fp32
        String engine = indexInfo.getKnnEngine().getName();

        String spaceType = indexInfo.getParameters().get(KNNConstants.SPACE_TYPE).toString();

        Map<String, Object> algorithmParams = new HashMap<>();
        algorithmParams.put("ef_construction", 100);
        algorithmParams.put("m", 16);

        Map<String, Object> indexParameters = new HashMap<>();
        indexParameters.put("algorithm", "hnsw");
        indexParameters.put("algorithm_parameters", algorithmParams);

        return RemoteBuildRequest.builder()
            .repositoryType(repositoryType)
            .containerName(containerName)
            .vectorPath(vectorPath)
            .docIdPath(docIdPath)
            .tenantId(tenantId)
            .dimension(dimension)
            .docCount(docCount)
            .dataType(dataType)
            .engine(engine)
            .indexParameters(indexParameters)
            .build();
    }

    /**
     * Submit vector build request to remote vector build service
     *
     */
    private void submitVectorBuild() {
        throw new NotImplementedException();
    }

    /**
     * Wait on remote vector build to complete
     */
    private String awaitVectorBuild(String jobId) {
        throw new NotImplementedException();
    }

    /**
     * Read constructed vector file from remote repository and write to IndexOutput
     */
    private void readFromRepository(String indexPath) {
        throw new NotImplementedException();
    }
}

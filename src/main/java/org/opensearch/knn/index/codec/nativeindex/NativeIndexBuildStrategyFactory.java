/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.Setter;
import org.apache.lucene.index.FieldInfo;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.common.featureflags.KNNFeatureFlags;
import org.opensearch.knn.index.codec.nativeindex.remote.RemoteIndexBuildStrategy;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.repositories.RepositoriesService;

import java.io.IOException;
import java.util.Optional;
import java.util.function.Supplier;

import static org.opensearch.knn.common.FieldInfoExtractor.extractKNNEngine;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;

/**
 * Creates the {@link NativeIndexBuildStrategy}
 */
public final class NativeIndexBuildStrategyFactory {

    private final Supplier<RepositoriesService> repositoriesServiceSupplier;
    private final IndexSettings indexSettings;
    @Setter
    private KNNMethodContext knnMethodContext;

    public NativeIndexBuildStrategyFactory() {
        this(null, null);
    }

    public NativeIndexBuildStrategyFactory(Supplier<RepositoriesService> repositoriesServiceSupplier, IndexSettings indexSettings) {
        this.repositoriesServiceSupplier = repositoriesServiceSupplier;
        this.indexSettings = indexSettings;
    }

    private Optional<KNNMethodContext> getKnnMethodContext() {
        return Optional.ofNullable(knnMethodContext);
    }

    /**
     * @param fieldInfo             Field related attributes/info
     * @param totalLiveDocs         Number of documents with the vector field. This values comes from {@link org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsWriter#flush}
     *                              and {@link org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsWriter#mergeOneField}
     * @param knnVectorValues       An instance of {@link KNNVectorValues} which is used to evaluate the size threshold KNN_REMOTE_VECTOR_BUILD_THRESHOLD
     * @return                      The {@link NativeIndexBuildStrategy} to be used. Intended to be used by {@link NativeIndexWriter}
     * @throws IOException
     */
    public NativeIndexBuildStrategy getBuildStrategy(
        final FieldInfo fieldInfo,
        final int totalLiveDocs,
        final KNNVectorValues<?> knnVectorValues
    ) throws IOException {
        final KNNEngine knnEngine = extractKNNEngine(fieldInfo);
        boolean isTemplate = fieldInfo.attributes().containsKey(MODEL_ID);
        boolean iterative = !isTemplate && KNNEngine.FAISS == knnEngine;

        NativeIndexBuildStrategy strategy = iterative
            ? MemOptimizedNativeIndexBuildStrategy.getInstance()
            : DefaultIndexBuildStrategy.getInstance();

        initializeVectorValues(knnVectorValues);
        long vectorBlobLength = ((long) knnVectorValues.bytesPerVector()) * totalLiveDocs;

        if (KNNFeatureFlags.isKNNRemoteVectorBuildEnabled()
            && repositoriesServiceSupplier != null
            && indexSettings != null
            && getKnnMethodContext().isPresent()
            && knnEngine.supportsRemoteIndexBuild(getKnnMethodContext().get().getMethodComponentContext())
            && RemoteIndexBuildStrategy.shouldBuildIndexRemotely(indexSettings, vectorBlobLength)) {
            return new RemoteIndexBuildStrategy(repositoriesServiceSupplier, strategy, indexSettings, getKnnMethodContext().get());
        } else {
            return strategy;
        }
    }
}

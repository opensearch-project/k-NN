/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import org.apache.lucene.index.FieldInfo;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.index.codec.nativeindex.remote.RemoteIndexBuildStrategy;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.repositories.RepositoriesService;

import java.util.function.Supplier;

import static org.opensearch.knn.common.FieldInfoExtractor.extractKNNEngine;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;

/**
 * Creates the {@link NativeIndexBuildStrategy}
 */
public final class NativeIndexBuildStrategyFactory {

    private final Supplier<RepositoriesService> repositoriesServiceSupplier;
    private final IndexSettings indexSettings;

    public NativeIndexBuildStrategyFactory() {
        this(null, null);
    }

    public NativeIndexBuildStrategyFactory(Supplier<RepositoriesService> repositoriesServiceSupplier, IndexSettings indexSettings) {
        this.repositoriesServiceSupplier = repositoriesServiceSupplier;
        this.indexSettings = indexSettings;
    }

    /**
     * Creates or returns the desired {@link NativeIndexBuildStrategy} implementation. Intended to be used by {@link NativeIndexWriter}
     * @param fieldInfo
     * @return
     */
    public NativeIndexBuildStrategy getBuildStrategy(final FieldInfo fieldInfo) {
        final KNNEngine knnEngine = extractKNNEngine(fieldInfo);
        boolean isTemplate = fieldInfo.attributes().containsKey(MODEL_ID);
        boolean iterative = !isTemplate && KNNEngine.FAISS == knnEngine;

        NativeIndexBuildStrategy strategy = iterative
            ? MemOptimizedNativeIndexBuildStrategy.getInstance()
            : DefaultIndexBuildStrategy.getInstance();

        if (repositoriesServiceSupplier != null
            && indexSettings != null
            && knnEngine.supportsRemoteIndexBuild()
            && RemoteIndexBuildStrategy.shouldBuildIndexRemotely(indexSettings)) {
            return new RemoteIndexBuildStrategy(repositoriesServiceSupplier, strategy);
        } else {
            return strategy;
        }
    }
}

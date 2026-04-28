/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import java.util.List;
import java.util.Map;

import org.opensearch.Version;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.knn.memoryoptsearch.VectorSearcherFactory;
import org.opensearch.remoteindexbuild.model.RemoteIndexParameters;

/**
 * Abstract base implementation of the {@link KNNEngine}.
 */
public abstract class AbstractKNNEngineBase implements KNNEngine {
    private final String name;

    public AbstractKNNEngineBase(String name) {
        this.name = name;
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public ValidationException validateMethod(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean isTrainingRequired(KNNMethodContext knnMethodContext) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int estimateOverheadInKB(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
        throw new UnsupportedOperationException();
    }

    @Override
    public String getExtension() {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean isRestricted(Version indexVersionCreated) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Version getRestrictedFromVersion() {
        throw new UnsupportedOperationException();
    }

    @Override
    public String getVersion() {
        throw new UnsupportedOperationException();
    }

    @Override
    public String getCompoundExtension() {
        throw new UnsupportedOperationException();
    }

    @Override
    public float score(float rawScore, SpaceType spaceType) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Float distanceToRadialThreshold(Float distance, SpaceType spaceType) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Float scoreToRadialThreshold(Float score, SpaceType spaceType) {
        throw new UnsupportedOperationException();
    }

    @Override
    public KNNLibraryIndexingContext getKNNLibraryIndexingContext(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        throw new UnsupportedOperationException();
    }

    @Override
    public KNNLibrarySearchContext getKNNLibrarySearchContext(String methodName) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Boolean isInitialized() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setInitialized(Boolean isInitialized) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<String> mmapFileExtensions() {
        throw new UnsupportedOperationException();
    }

    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        SpaceType spaceType
    ) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean supportsRemoteIndexBuild(KNNLibraryIndexingContext knnLibraryIndexingContext) {
        throw new UnsupportedOperationException();
    }

    @Override
    public RemoteIndexParameters createRemoteIndexingParameters(Map<String, Object> parameters) {
        throw new UnsupportedOperationException();
    }

    @Override
    public VectorSearcherFactory getVectorSearcherFactory() {
        throw new UnsupportedOperationException();
    }

    @Override
    public RescoreContext getRescoreContext(
        CompressionLevel compression,
        Mode mode,
        int dimension,
        Version version,
        boolean isFlatMethod,
        boolean isSQOneBit
    ) {
        throw new UnsupportedOperationException();
    }
}

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
 * Abstract interface to represent the SPI of the generic vector engine that is available
 * to the plugin.
 */
public interface KNNEngine {
    /**
     * The placeholder for undefined engine
     */
    final KNNEngine UNDEFINED = new AbstractKNNEngineBase("undefined") {
    };

    /**
     * Validates that the KNN method is compatible with the KNN engine.
     * @param knnMethodContext KNN method context
     * @param knnMethodConfigContext KNN method configuration context
     * @return an {@link ValidationException} instance on case of validation failure, otherwise {@code null}
     */
    ValidationException validateMethod(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext);

    /**
     * Checks if training is required or not
     * @param knnMethodContext KNN method context
     * @return {@code true} if training is required, {@code false} otherwise
     */
    boolean isTrainingRequired(KNNMethodContext knnMethodContext);

    /**
     * Estimates the overhead the KNN method adds irrespective of the number of vectors
     * @param knnMethodContext KNN method context
     * @param knnMethodConfigContext KNN method configuration context
     * @return size in Kilobytes
     */
    int estimateOverheadInKB(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext);

    /**
     * Returns this KNN engine name
     * @return engine name
     */
    String getName();

    /**
     * Gets the extension that files written with this KNN engine should have
     *
     * @return extension
     */
    String getExtension();

    /**
     * Checks if the KNN engine is deprecated for a given OpenSearch version.
     *
     * @param indexVersionCreated the OpenSearch version where the index is created
     * @return {@code true} if deprecated, {@code false} otherwise
     */
    boolean isRestricted(Version indexVersionCreated);

    /**
     * Gets the deprecated version
     *
     * @return deprecated vsersion
     */
    Version getRestrictedFromVersion();

    /**
     * Gets the version of the KNN engine that is being used. In general, this can be used for ensuring compatibility of
     * serialized artifacts. For instance, this can be used to check if a given file that was created on a different
     * cluster is compatible with this instance of the KNN engine.
     *
     * @return the string representing the KNN engine's version
     */
    String getVersion();

    /**
     * Gets the compound extension that files written with this KNN engine should have
     *
     * @return compound extension
     */
    String getCompoundExtension();

    /**
     * Generate the Lucene score from the rawScore returned by the KNN engine. With k-NN, often times the engine
     * will return a score where the lower the score, the better the result. This is the opposite of how Lucene scores
     * documents.
     *
     * @param rawScore  returned by the KNN engine
     * @param spaceType spaceType used to compute the score
     * @return Lucene score for the rawScore
     */
    float score(float rawScore, SpaceType spaceType);

    /**
     * Translates the distance radius input from end user to the engine's threshold.
     *
     * @param distance distance radius input from end user
     * @param spaceType spaceType used to compute the radius
     *
     * @return transformed distance for the KNN engine
     */
    Float distanceToRadialThreshold(Float distance, SpaceType spaceType);

    /**
     * Translates the score threshold input from end user to the engine's threshold.
     *
     * @param score score threshold input from end user
     * @param spaceType spaceType used to compute the threshold
     *
     * @return transformed score for the KNN engine
     */
    Float scoreToRadialThreshold(Float score, SpaceType spaceType);

    /**
     * Gets the context from the KNN engine needed to build the index.
     *
     * @param knnMethodContext KNN method context to get build context for
     * @param knnMethodConfigContext configuration context for the KNN method
     * @return parameter map
     */
    KNNLibraryIndexingContext getKNNLibraryIndexingContext(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext
    );

    /**
     * Gets metadata related to methods supported by the KNN engine
     *
     * @param methodName name of method
     * @return {@link KNNLibrarySearchContext} instance
     */
    KNNLibrarySearchContext getKNNLibrarySearchContext(String methodName);

    /**
     * Returns if KNN engine has been initialized or not
     *
     * @return whether KNN engine has been initialized
     */
    Boolean isInitialized();

    /**
     * Sets initialized to true
     *
     * @param isInitialized whether KNN engine has been initialized
     */
    void setInitialized(Boolean isInitialized);

    /**
     * Gets mmap file extensions
     *
     * @return list of file extensions that will be read/write with mmap
     */
    List<String> mmapFileExtensions();

    /**
     * Creates a new {@link ResolvedMethodContext} filling parameters based on other configuration details. A validation
     * exception will be thrown if the {@link KNNMethodConfigContext} is not compatible with the
     * parameters provided by the user.
     *
     * @param knnMethodContext User provided information regarding the method context. A new context should be
     *                         constructed. This variable will not be modified.
     * @param knnMethodConfigContext Configuration details that can be used for resolving the defaults. Should not be null
     * @param shouldRequireTraining Should the provided context require training
     * @param spaceType Space type for the method. Cannot be null or undefined
     * @return {@link ResolvedMethodContext} with dynamic defaults configured. This will include both the resolved
     *                                       compression as well as the completely resolve {@link KNNMethodContext}.
     *                                       This is guanteed to be a copy of the user provided context.
     * @throws org.opensearch.common.ValidationException on invalid configuration and user-provided context.
     */
    ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        final SpaceType spaceType
    );

    /**
     * Returns whether the engine implementation supports remote index build
     * @return {@code true} if remote index build is supported, {@code false} otherwise
     */
    boolean supportsRemoteIndexBuild(KNNLibraryIndexingContext knnLibraryIndexingContext);

    /**
     * Creates the set of index parameters needed to build the remote index
     * @param parameters parameters
     * @return {@link RemoteIndexParameters} instance
     */
    RemoteIndexParameters createRemoteIndexingParameters(Map<String, Object> parameters);

    /**
     * Create a new vector searcher factory that compatible with on Lucene search API.
     * @return New searcher factory that returns {@link org.opensearch.knn.memoryoptsearch.VectorSearcher}
     *         If it is not supported, it should return null.
     *         But, if it is supported, the factory shall not return null searcher.
     */
    VectorSearcherFactory getVectorSearcherFactory();

    /**
     * Returns {@link RescoreContext} that is specific to the KNN engine
     * @param compression compression level
     * @param mode mode
     * @param dimension vector dimensions
     * @param version OpenSearch version
     * @param isFlatMethod flat method or not
     * @param isSQOneBit SQ one bit or not
     * @return {@link RescoreContext} that is specific to the KNN engine, {@code null} otherwise
     */
    RescoreContext getRescoreContext(
        CompressionLevel compression,
        Mode mode,
        int dimension,
        Version version,
        boolean isFlatMethod,
        boolean isSQOneBit
    );
}

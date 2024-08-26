/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.SpaceType;

import java.util.Collections;
import java.util.List;

/**
 * KNNLibrary is an interface that helps the plugin communicate with k-NN libraries
 */
public interface KNNLibrary {

    /**
     * Gets the version of the library that is being used. In general, this can be used for ensuring compatibility of
     * serialized artifacts. For instance, this can be used to check if a given file that was created on a different
     * cluster is compatible with this instance of the library.
     *
     * @return the string representing the library's version
     */
    String getVersion();

    /**
     * Gets the extension that files written with this library should have
     *
     * @return extension
     */
    String getExtension();

    /**
     * Gets the compound extension that files written with this library should have
     *
     * @return compound extension
     */
    String getCompoundExtension();

    /**
     * Generate the Lucene score from the rawScore returned by the library. With k-NN, often times the library
     * will return a score where the lower the score, the better the result. This is the opposite of how Lucene scores
     * documents.
     *
     * @param rawScore  returned by the library
     * @param spaceType spaceType used to compute the score
     * @return Lucene score for the rawScore
     */
    float score(float rawScore, SpaceType spaceType);

    /**
     * Translate the distance radius input from end user to the engine's threshold.
     *
     * @param distance distance radius input from end user
     * @param spaceType spaceType used to compute the radius
     *
     * @return transformed distance for the library
     */
    Float distanceToRadialThreshold(Float distance, SpaceType spaceType);

    /**
     * Translate the score threshold input from end user to the engine's threshold.
     *
     * @param score score threshold input from end user
     * @param spaceType spaceType used to compute the threshold
     *
     * @return transformed score for the library
     */
    Float scoreToRadialThreshold(Float score, SpaceType spaceType);

    /**
     * Validate the knnMethodContext for the given library. A ValidationException should be thrown if the method is
     * deemed invalid.
     *
     * @param knnMethodConfigContext configuration context for the method
     * @return ValidationException produced by validation errors; null if no validations errors.
     */
    ValidationException validateMethod(KNNMethodConfigContext knnMethodConfigContext);

    /**
     * Returns whether training is required or not from knnMethodConfigContext for the given library.
     *
     * @param knnMethodConfigContext methodContext
     * @return true if training is required; false otherwise
     */
    boolean isTrainingRequired(KNNMethodConfigContext knnMethodConfigContext);

    /**
     * Get the context from the library needed to build the index.
     *
     * @param knnMethodConfigContext configuration context for the method
     * @return parameter map
     */
    KNNLibraryIndexingContext getKNNLibraryIndexingContext(KNNMethodConfigContext knnMethodConfigContext);

    /**
     * Getter for initialized
     *
     * @return whether library has been initialized
     */
    Boolean isInitialized();

    /**
     * Set initialized to true
     *
     * @param isInitialized whether library has been initialized
     */
    void setInitialized(Boolean isInitialized);

    /**
     * Getter for mmap file extensions
     *
     * @return list of file extensions that will be read/write with mmap
     */
    default List<String> mmapFileExtensions() {
        return Collections.emptyList();
    }
}

/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.util;

import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.KNNMethod;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.training.VectorSpaceInfo;

import java.util.Collections;
import java.util.List;
import java.util.Map;

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
     * Gets a particular KNN method that the library supports. This should throw an exception if the method is not
     * supported by the library.
     *
     * @param methodName name of the method to be looked up
     * @return KNNMethod in the library corresponding to the method name
     */
    KNNMethod getMethod(String methodName);

    /**
     * Gets metadata related to methods supported by the library
     * @param methodName
     * @return
     */
    EngineSpecificMethodContext getMethodContext(String methodName);

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
     * @param knnMethodContext to be validated
     * @return ValidationException produced by validation errors; null if no validations errors.
     */
    ValidationException validateMethod(KNNMethodContext knnMethodContext);

    /**
     * Validate the knnMethodContext for the given library, using additional data not present in the method context. A ValidationException should be thrown if the method is
     * deemed invalid.
     *
     * @param knnMethodContext to be validated
     * @param vectorSpaceInfo additional data not present in the method context
     * @return ValidationException produced by validation errors; null if no validations errors.
     */
    ValidationException validateMethodWithData(KNNMethodContext knnMethodContext, VectorSpaceInfo vectorSpaceInfo);

    /**
     * Returns whether training is required or not from knnMethodContext for the given library.
     *
     * @param knnMethodContext methodContext
     * @return true if training is required; false otherwise
     */
    boolean isTrainingRequired(KNNMethodContext knnMethodContext);

    /**
     * Estimate overhead of KNNMethodContext in Kilobytes.
     *
     * @param knnMethodContext to estimate size for
     * @param dimension        to estimate size for
     * @return size overhead estimate in KB
     */
    int estimateOverheadInKB(KNNMethodContext knnMethodContext, int dimension);

    /**
     * Generate method as map that can be used to configure the knn index from the jni
     *
     * @param knnMethodContext to generate parameter map from
     * @return parameter map
     */
    Map<String, Object> getMethodAsMap(KNNMethodContext knnMethodContext);

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

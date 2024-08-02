/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.training.VectorSpaceInfo;

/**
 * KNNMethod defines the structure of a method supported by a particular k-NN library. It is used to validate
 * the KNNMethodContext passed in by the user, where the KNNMethodContext provides the configuration that the user may
 * want. Then, it provides the information necessary to build and search engine knn indices.
 */
public interface KNNMethod {

    /**
     * Determines whether the provided space is supported for this method
     *
     * @param space to be checked
     * @return true if the space is supported; false otherwise
     */
    boolean isSpaceTypeSupported(SpaceType space);

    /**
     * Validate that the configured KNNMethodContext is valid for this method
     *
     * @param knnMethodContext to be validated
     * @return ValidationException produced by validation errors; null if no validations errors.
     */
    ValidationException validate(KNNMethodContext knnMethodContext);

    /**
     * Validate that the configured KNNMethodContext is valid for this method, using additional data not present in the method context
     *
     * @param knnMethodContext to be validated
     * @param vectorSpaceInfo additional data not present in the method context
     * @return ValidationException produced by validation errors; null if no validations errors.
     */
    ValidationException validateWithData(KNNMethodContext knnMethodContext, VectorSpaceInfo vectorSpaceInfo);

    /**
     * returns whether training is required or not
     *
     * @param knnMethodContext context to check if training is required on
     * @return true if training is required; false otherwise
     */
    boolean isTrainingRequired(KNNMethodContext knnMethodContext);

    /**
     * Returns the estimated overhead of the method in KB
     *
     * @param knnMethodContext context to estimate overhead
     * @param dimension dimension to make estimate with
     * @return estimate overhead in KB
     */
    int estimateOverheadInKB(KNNMethodContext knnMethodContext, int dimension);

    /**
     * Parse knnMethodContext into context that the library can use to build the index
     *
     * @param knnMethodContext to generate the context for
     * @return KNNLibraryIndexBuildContext
     */
    KNNLibraryIndexBuildContext getKNNLibraryIndexBuildContext(KNNMethodContext knnMethodContext);

    /**
     * Get the search context for a particular method
     *
     * @return KNNLibrarySearchContext
     */
    KNNLibrarySearchContext getKNNLibrarySearchContext();
}

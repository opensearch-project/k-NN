/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.common.ValidationException;

/**
 * KNNMethod defines the structure of a method supported by a particular k-NN library. It is used to validate
 * the KNNMethodContext passed in by the user, where the KNNMethodContext provides the configuration that the user may
 * want. Then, it provides the information necessary to build and search engine knn indices.
 */
public interface KNNMethod {
    /**
     * Validate that the configured KNNMethodContext is valid for this method
     *
     * @param knnMethodConfigContext to be validated
     * @return ValidationException produced by validation errors; null if no validations errors.
     */
    ValidationException validate(KNNMethodConfigContext knnMethodConfigContext);

    /**
     * Parse knnMethodContext into context that the library can use to build the index
     *
     * @param knnMethodConfigContext to generate the context for
     * @return KNNLibraryIndexingContext
     */
    KNNLibraryIndexingContext getKNNLibraryIndexingContext(KNNMethodConfigContext knnMethodConfigContext);

    /**
     * returns whether training is required or not
     *
     * @param knnMethodConfigContext context to check if training is required on
     * @return true if training is required; false otherwise
     */
    boolean isTrainingRequired(KNNMethodConfigContext knnMethodConfigContext);
}

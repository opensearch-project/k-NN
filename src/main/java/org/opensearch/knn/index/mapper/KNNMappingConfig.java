/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.knn.index.engine.KNNMethodContext;

import java.util.Optional;

/**
 * Class holds information about how the ANN indices are created. The design of this class ensures that we do not
 * accidentally configure an index that has multiple ways it can be created. This class is immutable.
 */
public interface KNNMappingConfig {
    /**
     *
     * @return Optional containing the modelId if created from model, otherwise empty
     */
    default Optional<String> getModelId() {
        return Optional.empty();
    }

    /**
     *
     * @return Optional containing the KNNMethodContext if created from method, otherwise empty
     */
    default Optional<KNNMethodContext> getKnnMethodContext() {
        return Optional.empty();
    }

    /**
     *
     * @return the dimension of the index; for model based indices, it will be null
     */
    int getDimension();
}

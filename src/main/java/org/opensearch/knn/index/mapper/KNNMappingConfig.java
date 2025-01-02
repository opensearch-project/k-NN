/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.Version;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;

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
     * Return the mode to be used for this field
     *
     * @return {@link Mode}
     */
    default Mode getMode() {
        return Mode.NOT_CONFIGURED;
    }

    /**
     * Return compression level to be used for this field
     *
     * @return {@link CompressionLevel}
     */
    default CompressionLevel getCompressionLevel() {
        return CompressionLevel.NOT_CONFIGURED;
    }

    /**
     * Returns quantization config
     * @return
     */
    default QuantizationConfig getQuantizationConfig() {
        return QuantizationConfig.EMPTY;
    }

    /**
     *
     * @return the dimension of the index; for model based indices, it will be null
     */
    int getDimension();

    /**
     * Returns index created Version
     * @return Version
     */
    default Version getIndexCreatedVersion() {
        return Version.CURRENT;
    }
}

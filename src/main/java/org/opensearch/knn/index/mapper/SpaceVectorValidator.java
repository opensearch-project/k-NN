/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.AllArgsConstructor;
import org.opensearch.knn.index.SpaceType;

/**
 * Confirms that a given vector is valid for the provided space type
 */
@AllArgsConstructor
public class SpaceVectorValidator implements VectorValidator {

    private final SpaceType spaceType;

    @Override
    public void validateVector(byte[] vector) {
        spaceType.validateVector(vector);
    }

    @Override
    public void validateVector(float[] vector) {
        spaceType.validateVector(vector);
    }
}

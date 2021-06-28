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

package org.opensearch.knn.indices;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;

import java.util.Objects;

public class Model {
    final private KNNEngine knnEngine;
    final private SpaceType spaceType;
    final private byte[] modelBlob;

    public Model(KNNEngine knnEngine, SpaceType spaceType, byte[] modelBlob) {
        this.knnEngine = Objects.requireNonNull(knnEngine, "knnEngine must not be null");
        this.spaceType = Objects.requireNonNull(spaceType, "spaceType must not be null");
        this.modelBlob = Objects.requireNonNull(modelBlob, "modelBlob must not be null");
    }

    public KNNEngine getKnnEngine() {
        return knnEngine;
    }

    public SpaceType getSpaceType() {
        return spaceType;
    }

    public byte[] getModelBlob() {
        return modelBlob;
    }
}

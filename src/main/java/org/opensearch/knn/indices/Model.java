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

import org.apache.commons.lang.builder.EqualsBuilder;
import org.apache.commons.lang.builder.HashCodeBuilder;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;

import java.util.Objects;

import static org.opensearch.knn.index.KNNVectorFieldMapper.MAX_DIMENSION;

public class Model {

    final private KNNEngine knnEngine;
    final private SpaceType spaceType;
    final private int dimension;
    final private byte[] modelBlob;

    /**
     * Constructor
     *
     * @param knnEngine engine model is built with
     * @param spaceType space type model uses
     * @param modelBlob binary representation of model template index
     */
    public Model(KNNEngine knnEngine, SpaceType spaceType, int dimension, byte[] modelBlob) {
        this.knnEngine = Objects.requireNonNull(knnEngine, "knnEngine must not be null");
        this.spaceType = Objects.requireNonNull(spaceType, "spaceType must not be null");
        if (dimension <= 0 || dimension >= MAX_DIMENSION) {
            throw new IllegalArgumentException("Dimension \"" + dimension + "\" is invalid. Value must be greater " +
                    "than 0 and less than " + MAX_DIMENSION);
        }
        this.dimension = dimension;
        this.modelBlob = Objects.requireNonNull(modelBlob, "modelBlob must not be null");
    }

    /**
     * getter for model's knnEngine
     *
     * @return knnEngine
     */
    public KNNEngine getKnnEngine() {
        return knnEngine;
    }

    /**
     * getter for model's spaceType
     *
     * @return spaceType
     */
    public SpaceType getSpaceType() {
        return spaceType;
    }

    /**
     * getter for model's dimension
     *
     * @return dimension
     */
    public int getDimension() {
        return dimension;
    }

    /**
     * getter for model's binary blob
     *
     * @return modelBlob
     */
    public byte[] getModelBlob() {
        return modelBlob;
    }

    /**
     * getter for model's length
     *
     * @return length of model blob
     */
    public int getLength() {
        return modelBlob.length;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null || getClass() != obj.getClass())
            return false;
        Model other = (Model) obj;

        EqualsBuilder equalsBuilder = new EqualsBuilder();
        equalsBuilder.append(knnEngine, other.knnEngine);
        equalsBuilder.append(spaceType, other.spaceType);
        equalsBuilder.append(dimension, other.dimension);
        equalsBuilder.append(modelBlob, other.modelBlob);

        return equalsBuilder.isEquals();
    }

    @Override
    public int hashCode() {
        return new HashCodeBuilder().append(knnEngine).append(spaceType).append(dimension).append(modelBlob).toHashCode();
    }
}

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
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.io.stream.Writeable;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;
import java.util.Objects;

import static org.opensearch.knn.index.KNNVectorFieldMapper.MAX_DIMENSION;

public class ModelInfo implements Writeable {

    public static final String MODEL_INFO_FIELD = "knn-models";
    private static final String DELIMITER = ",";

    final private KNNEngine knnEngine;
    final private SpaceType spaceType;
    final private int dimension;

    /**
     * Constructor
     *
     * @param in Stream input
     */
    public ModelInfo(StreamInput in) throws IOException {
        this.knnEngine = KNNEngine.getEngine(in.readString());
        this.spaceType = SpaceType.getSpace(in.readString());
        this.dimension = in.readInt();
    }

    /**
     * Constructor
     *
     * @param knnEngine engine model is built with
     * @param spaceType space type model uses
     * @param dimension dimension of the model
     */
    public ModelInfo(KNNEngine knnEngine, SpaceType spaceType, int dimension) {
        this.knnEngine = Objects.requireNonNull(knnEngine, "knnEngine must not be null");
        this.spaceType = Objects.requireNonNull(spaceType, "spaceType must not be null");
        if (dimension <= 0 || dimension >= MAX_DIMENSION) {
            throw new IllegalArgumentException("Dimension \"" + dimension + "\" is invalid. Value must be greater " +
                    "than 0 and less than " + MAX_DIMENSION);
        }
        this.dimension = dimension;
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

    @Override
    public String toString() {
        return knnEngine.getName() + DELIMITER + spaceType.getValue() + DELIMITER + dimension;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null || getClass() != obj.getClass())
            return false;
        ModelInfo other = (ModelInfo) obj;

        EqualsBuilder equalsBuilder = new EqualsBuilder();
        equalsBuilder.append(knnEngine, other.knnEngine);
        equalsBuilder.append(spaceType, other.spaceType);
        equalsBuilder.append(dimension, other.dimension);

        return equalsBuilder.isEquals();
    }

    @Override
    public int hashCode() {
        return new HashCodeBuilder().append(knnEngine).append(spaceType).append(dimension).toHashCode();
    }

    /**
     * Returns ModelInfo from string representation
     *
     * @param modelInfoString String to be parsed
     * @return ModelInfo from string
     */
    public static ModelInfo fromString(String modelInfoString) {
        String[] modelInfoArray = modelInfoString.split(DELIMITER);

        if (modelInfoArray.length != 3) {
            throw new IllegalArgumentException("Illegal format for model info. Must be of the form " +
                    "\"<KNNEngine>,<SpaceType>,<Dimension>\"");
        }

        KNNEngine knnEngine = KNNEngine.getEngine(modelInfoArray[0]);
        SpaceType spaceType = SpaceType.getSpace(modelInfoArray[1]);
        int dimension = Integer.parseInt(modelInfoArray[2]);

        return new ModelInfo(knnEngine, spaceType, dimension);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeString(knnEngine.getName());
        out.writeString(spaceType.getValue());
        out.writeInt(dimension);
    }
}

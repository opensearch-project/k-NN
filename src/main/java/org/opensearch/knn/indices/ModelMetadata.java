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
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.common.io.stream.Writeable;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.IndexUtil;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

import static org.opensearch.knn.common.KNNConstants.*;

public class ModelMetadata implements Writeable, ToXContentObject {

    private static final String DELIMITER = ",";

    final private KNNEngine knnEngine;
    final private SpaceType spaceType;
    final private int dimension;

    private AtomicReference<ModelState> state;
    final private String timestamp;
    final private String description;
    private String error;
    private String nodeAssignment;

    /**
     * Constructor
     *
     * @param in Stream input
     */
    public ModelMetadata(StreamInput in) throws IOException {
        this.knnEngine = KNNEngine.getEngine(in.readString());
        this.spaceType = SpaceType.getSpace(in.readString());
        this.dimension = in.readInt();
        this.state = new AtomicReference<>(ModelState.readFrom(in));
        this.timestamp = in.readString();

        // Description and error may be empty. However, reading the string will work as long as they are not null
        // which is checked in constructor and setters
        this.description = in.readString();
        this.error = in.readString();
        this.nodeAssignment = in.readOptionalString();
    }

    /**
     * Constructor
     *
     * @param knnEngine engine model is built with
     * @param spaceType space type model uses
     * @param dimension dimension of the model
     * @param modelState state of the model
     * @param timestamp timevalue when model was created
     * @param description information about the model
     * @param error error message associated with model
     */
    public ModelMetadata(
        KNNEngine knnEngine,
        SpaceType spaceType,
        int dimension,
        ModelState modelState,
        String timestamp,
        String description,
        String error,
        String nodeAssignment
    ) {
        this.knnEngine = Objects.requireNonNull(knnEngine, "knnEngine must not be null");
        this.spaceType = Objects.requireNonNull(spaceType, "spaceType must not be null");
        int maxDimensions = KNNEngine.getMaxDimensionByEngine(this.knnEngine);
        if (dimension <= 0 || dimension > maxDimensions) {
            throw new IllegalArgumentException(
                String.format(
                    "Dimension \"%s\" is invalid. Value must be greater than 0 and less than or equal to %d",
                    dimension,
                    maxDimensions
                )
            );
        }
        this.dimension = dimension;

        this.state = new AtomicReference<>(Objects.requireNonNull(modelState, "modelState must not be null"));
        this.timestamp = Objects.requireNonNull(timestamp, "timestamp must not be null");
        this.description = Objects.requireNonNull(description, "description must not be null");
        this.error = Objects.requireNonNull(error, "error must not be null");
        this.nodeAssignment = nodeAssignment;
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
     * getter for model's state
     *
     * @return state
     */
    public ModelState getState() {
        return state.get();
    }

    /**
     * getter for model's timestamp
     *
     * @return timestamp
     */
    public String getTimestamp() {
        return timestamp;
    }

    /**
     * getter for model's description
     *
     * @return description
     */
    public String getDescription() {
        return description;
    }

    /**
     * getter for model's error
     *
     * @return error
     */
    public String getError() {
        return error;
    }

    /**
     * getter for model's node assignment
     *
     * @return nodeAssignment
     */
    public String getNodeAssignment() {
        return nodeAssignment;
    }

    public void setNodeAssignment(String nodeAssignment) {
        this.nodeAssignment = nodeAssignment;
    }

    /**
     * setter for model's state
     *
     * @param state of the model
     */
    public synchronized void setState(ModelState state) {
        this.state.set(Objects.requireNonNull(state, "state must not be null"));
    }

    /**
     * setter for model's error
     *
     * @param error set on failure
     */
    public synchronized void setError(String error) {
        this.error = error;
    }

    @Override
    public String toString() {
        return String.join(
            DELIMITER,
            knnEngine.getName(),
            spaceType.getValue(),
            Integer.toString(dimension),
            getState().toString(),
            timestamp,
            description,
            error,
            nodeAssignment
        );
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        ModelMetadata other = (ModelMetadata) obj;

        EqualsBuilder equalsBuilder = new EqualsBuilder();
        equalsBuilder.append(getKnnEngine(), other.getKnnEngine());
        equalsBuilder.append(getSpaceType(), other.getSpaceType());
        equalsBuilder.append(getDimension(), other.getDimension());
        equalsBuilder.append(getState(), other.getState());
        equalsBuilder.append(getTimestamp(), other.getTimestamp());
        equalsBuilder.append(getDescription(), other.getDescription());
        equalsBuilder.append(getError(), other.getError());

        return equalsBuilder.isEquals();
    }

    @Override
    public int hashCode() {
        return new HashCodeBuilder().append(getKnnEngine())
            .append(getSpaceType())
            .append(getDimension())
            .append(getState())
            .append(getTimestamp())
            .append(getDescription())
            .append(getError())
            .toHashCode();
    }

    /**
     * Returns ModelMetadata from string representation
     *
     * @param modelMetadataString String to be parsed
     * @return modelMetadata from string
     */
    public static ModelMetadata fromString(String modelMetadataString) {
        String[] modelMetadataArray = modelMetadataString.split(DELIMITER, -1);

        if (modelMetadataArray.length != 8) {
            if (IndexUtil.isClusterOnOrAfterMinRequiredVersion("model_node_assignment")) {
                throw new IllegalArgumentException(
                        "Illegal format for model metadata. Must be of the form "
                                + "\"<KNNEngine>,<SpaceType>,<Dimension>,<ModelState>,<Timestamp>,<Description>,<Error>,<NodeAssignment>\"."
                );
            } else if (modelMetadataArray.length != 7) {
                throw new IllegalArgumentException(
                        "Illegal format for model metadata. Must be of the form "
                                + "\"<KNNEngine>,<SpaceType>,<Dimension>,<ModelState>,<Timestamp>,<Description>,<Error>\"."
                );
            }
        }

        KNNEngine knnEngine = KNNEngine.getEngine(modelMetadataArray[0]);
        SpaceType spaceType = SpaceType.getSpace(modelMetadataArray[1]);
        int dimension = Integer.parseInt(modelMetadataArray[2]);
        ModelState modelState = ModelState.getModelState(modelMetadataArray[3]);
        String timestamp = modelMetadataArray[4];
        String description = modelMetadataArray[5];
        String error = modelMetadataArray[6];
        if (IndexUtil.isClusterOnOrAfterMinRequiredVersion("model_node_assignment")) {
            String nodeAssignment = modelMetadataArray[7];
            return new ModelMetadata(knnEngine, spaceType, dimension, modelState, timestamp, description, error, nodeAssignment);
        }
        return new ModelMetadata(knnEngine, spaceType, dimension, modelState, timestamp, description, error, "");
    }

    private static String objectToString(Object value) {
        if (value == null) return null;
        return (String) value;
    }

    private static Integer objectToInteger(Object value) {
        if (value == null) return null;
        return (Integer) value;
    }

    /**
     * Returns ModelMetadata from Map representation
     *
     * @param modelSourceMap Map to be parsed
     * @return ModelMetadata instance
     */
    public static ModelMetadata getMetadataFromSourceMap(final Map<String, Object> modelSourceMap) {
        Object engine = modelSourceMap.get(KNNConstants.KNN_ENGINE);
        Object space = modelSourceMap.get(KNNConstants.METHOD_PARAMETER_SPACE_TYPE);
        Object dimension = modelSourceMap.get(KNNConstants.DIMENSION);
        Object state = modelSourceMap.get(KNNConstants.MODEL_STATE);
        Object timestamp = modelSourceMap.get(KNNConstants.MODEL_TIMESTAMP);
        Object description = modelSourceMap.get(KNNConstants.MODEL_DESCRIPTION);
        Object error = modelSourceMap.get(KNNConstants.MODEL_ERROR);
        Object nodeAssignment = modelSourceMap.get(KNNConstants.MODEL_NODE_ASSIGNMENT);

        ModelMetadata modelMetadata = new ModelMetadata(
            KNNEngine.getEngine(objectToString(engine)),
            SpaceType.getSpace(objectToString(space)),
            objectToInteger(dimension),
            ModelState.getModelState(objectToString(state)),
            objectToString(timestamp),
            objectToString(description),
            objectToString(error),
            objectToString(nodeAssignment)
        );
        return modelMetadata;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeString(getKnnEngine().getName());
        out.writeString(getSpaceType().getValue());
        out.writeInt(getDimension());
        getState().writeTo(out);
        out.writeString(getTimestamp());
        out.writeString(getDescription());
        out.writeString(getError());
        if (IndexUtil.isClusterOnOrAfterMinRequiredVersion("model_node_assignment")) {
            out.writeString(getNodeAssignment());
        }
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.field(MODEL_STATE, getState().getName());
        builder.field(MODEL_TIMESTAMP, getTimestamp());
        builder.field(MODEL_DESCRIPTION, getDescription());
        builder.field(MODEL_ERROR, getError());

        builder.field(METHOD_PARAMETER_SPACE_TYPE, getSpaceType().getValue());
        builder.field(DIMENSION, getDimension());
        builder.field(KNN_ENGINE, getKnnEngine().getName());
        if (IndexUtil.isClusterOnOrAfterMinRequiredVersion("model_node_assignment")) {
            builder.field(MODEL_NODE_ASSIGNMENT, getNodeAssignment());
        }
        return builder;
    }
}

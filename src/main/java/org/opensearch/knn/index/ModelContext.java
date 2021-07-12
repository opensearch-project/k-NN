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

package org.opensearch.knn.index;

import org.apache.commons.lang.builder.EqualsBuilder;
import org.apache.commons.lang.builder.HashCodeBuilder;
import org.opensearch.common.xcontent.ToXContent;
import org.opensearch.common.xcontent.ToXContentFragment;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;

/**
 * Used to capture context of a model
 */
public class ModelContext implements ToXContentFragment {

    private final String modelId;
    private final KNNEngine knnEngine;
    private final SpaceType spaceType;
    private final int dimension;

    /**
     * Constructor
     *
     * @param modelId Identifier for model
     * @param knnEngine KNNEngine of model
     * @param spaceType SpaceType of model
     * @param dimension Dimension of model
     */
    public ModelContext(String modelId, KNNEngine knnEngine, SpaceType spaceType, int dimension) {
        this.modelId = modelId;
        this.knnEngine = knnEngine;
        this.spaceType = spaceType;
        this.dimension = dimension;
    }

    /**
     * Getter for model's id
     *
     * @return model's id
     */
    public String getModelId() {
        return modelId;
    }

    /**
     * Getter for model's engine
     *
     * @return model's engine
     */
    public KNNEngine getKNNEngine() {
        return knnEngine;
    }

    /**
     * Getter for model's space type
     *
     * @return model's space type
     */
    public SpaceType getSpaceType() {
        return spaceType;
    }

    /**
     * Getter for model's dimension
     *
     * @return model's dimension
     */
    public int getDimension() {
        return dimension;
    }

    /**
     * Parse an object to a ModelContext
     *
     * @param in String of model id
     * @return ModelContext constructed from model identified by in
     */
    public static ModelContext parse(Object in) {
        if (!(in instanceof Map)) {
            throw new MapperParsingException("Unable to parse ModelContext: provided input is not of type \"Map\"");
        }

        @SuppressWarnings("unchecked")
        Map<String, Object> methodMap = (Map<String, Object>) in;

        String modelId = null;
        KNNEngine knnEngine = null;
        SpaceType spaceType = null;
        Integer dimension = null;

        String key;
        Object value;
        for (Map.Entry<String, Object> methodEntry : methodMap.entrySet()) {
            key = methodEntry.getKey();
            value = methodEntry.getValue();

            if (MODEL_ID.equals(key)) {
                if (!(value instanceof String)) {
                    throw new MapperParsingException("\"" + MODEL_ID + "\" must be a string");
                }

                modelId = (String) value;
            } else if (KNN_ENGINE.equals(key)) {
                if (!(value instanceof String)) {
                    throw new MapperParsingException("\"" + KNN_ENGINE + "\" must be a string");
                }

                knnEngine = KNNEngine.getEngine((String) value);
            } else if (METHOD_PARAMETER_SPACE_TYPE.equals(key)) {
                if (!(value instanceof String)) {
                    throw new MapperParsingException("\"" + METHOD_PARAMETER_SPACE_TYPE + "\" must be a string");
                }

                spaceType = SpaceType.getSpace((String) value);
            } else if (DIMENSION.equals(key)) {
                if (!(value instanceof Integer)) {
                    throw new MapperParsingException("\"" + DIMENSION + "\" must be an integer");
                }

                dimension = (Integer) value;
            } else {
                throw new MapperParsingException("Invalid parameter: " + key);
            }
        }

        if (modelId == null) {
            throw new MapperParsingException("\"" + MODEL_ID + "\" must be set");
        }

        if (knnEngine == null) {
            throw new MapperParsingException("\"" + KNN_ENGINE + "\" must be set");
        }

        if (spaceType == null) {
            throw new MapperParsingException("\"" + METHOD_PARAMETER_SPACE_TYPE + "\" must be set");
        }

        if (dimension == null) {
            throw new MapperParsingException("\"" + DIMENSION + "\" must be set");
        }

        return new ModelContext(modelId, knnEngine, spaceType, dimension);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, ToXContent.Params params) throws IOException {
        builder.field(MODEL_ID, modelId);
        builder.field(KNN_ENGINE, knnEngine.getName());
        builder.field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue());
        builder.field(DIMENSION, dimension);
        return builder;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null || getClass() != obj.getClass())
            return false;
        ModelContext other = (ModelContext) obj;

        EqualsBuilder equalsBuilder = new EqualsBuilder();
        equalsBuilder.append(modelId, other.modelId);
        equalsBuilder.append(knnEngine, other.knnEngine);
        equalsBuilder.append(spaceType, other.spaceType);
        equalsBuilder.append(dimension, other.dimension);

        return equalsBuilder.isEquals();
    }

    @Override
    public int hashCode() {
        return new HashCodeBuilder().append(modelId).append(knnEngine).append(spaceType).append(dimension).toHashCode();
    }
}

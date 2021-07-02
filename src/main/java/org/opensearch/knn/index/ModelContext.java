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

import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelCache;

/**
 * Used to capture context of a model
 */
public class ModelContext {

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
        if (!(in instanceof String)) {
            throw new MapperParsingException("Unable to parse ModelContext: provided input is not of type \"String\"");
        }

        String modelId = (String) in;
        Model model = ModelCache.getInstance().get(modelId);
        KNNEngine knnEngine = model.getKnnEngine();
        SpaceType spaceType = model.getSpaceType();
        int dimension = model.getDimension();

        return new ModelContext(modelId, knnEngine, spaceType, dimension);
    }
}

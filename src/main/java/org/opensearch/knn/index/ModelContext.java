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
 * Used to provide context for a given model identifier. Context includes the engine the model is used for as well as
 * the space type the model uses
 */
public class ModelContext {

    private final String modelId;
    private final KNNEngine knnEngine;
    private final SpaceType spaceType;

    public ModelContext(String modelId) {
        this.modelId = modelId;
        Model model = ModelCache.getInstance().get(modelId);
        this.knnEngine = model.getKnnEngine();
        this.spaceType = model.getSpaceType();
    }

    public String getModelId() {
        return modelId;
    }

    public KNNEngine getKNNEngine() {
        return knnEngine;
    }

    public SpaceType getSpaceType() {
        return spaceType;
    }

    public static ModelContext parse(Object in) {
        if (!(in instanceof String)) {
            throw new MapperParsingException("Unable to parse ModelContext: provided input is not of type \"String\"");
        }

        return new ModelContext((String) in);
    }
}

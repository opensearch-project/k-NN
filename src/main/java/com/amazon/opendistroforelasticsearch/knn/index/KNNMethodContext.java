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

package com.amazon.opendistroforelasticsearch.knn.index;

import com.amazon.opendistroforelasticsearch.knn.index.util.KNNEngine;
import org.opensearch.common.xcontent.ToXContentFragment;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.index.mapper.MapperParsingException;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;

import static com.amazon.opendistroforelasticsearch.knn.common.KNNConstants.KNN_ENGINE;
import static com.amazon.opendistroforelasticsearch.knn.common.KNNConstants.METHOD_HNSW;
import static com.amazon.opendistroforelasticsearch.knn.common.KNNConstants.NAME;
import static com.amazon.opendistroforelasticsearch.knn.common.KNNConstants.PARAMETERS;
import static com.amazon.opendistroforelasticsearch.knn.common.KNNConstants.SPACE_TYPE;

/**
 * KNNMethodContext will contain the information necessary to produce a library index from an Opensearch mapping.
 * It will encompass all parameters necessary to build the index.
 */
public class KNNMethodContext implements ToXContentFragment {

    public static final KNNMethodContext DEFAULT = new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.DEFAULT,
            new MethodComponentContext(METHOD_HNSW, Collections.emptyMap()));

    private KNNEngine knnEngine;
    private SpaceType spaceType;
    private MethodComponentContext methodComponent;

    /**
     * Constructor
     *
     * @param knnEngine engine that this method uses
     * @param spaceType space type that this method uses
     * @param methodComponent MethodComponent describing the main index
     */
    public KNNMethodContext(KNNEngine knnEngine, SpaceType spaceType, MethodComponentContext methodComponent) {
        this.knnEngine = knnEngine;
        this.spaceType = spaceType;
        this.methodComponent = methodComponent;
    }

    /**
     * Gets the main method component
     *
     * @return methodComponent
     */
    public MethodComponentContext getMethodComponent() {
        return methodComponent;
    }

    /**
     * Gets the engine to be used for this context
     *
     * @return knnEngine
     */
    public KNNEngine getEngine() {
        return knnEngine;
    }

    /**
     * Gets the space type for this context
     *
     * @return spaceType
     */
    public SpaceType getSpaceType() {
        return spaceType;
    }

    /**
     * This method uses the knnEngine to validate that the method is compatible with the engine
     *
     */
    public void validate() {
        knnEngine.validateMethod(this);
    }

    /**
     * Parses an Object into a KNNMethodContext.
     *
     * @param in Object containing mapping to be parsed
     * @return KNNMethodContext
     */
    public static KNNMethodContext parse(Object in) {
        if (!(in instanceof Map)) {
            throw new MapperParsingException("Unable to parse mapping into KNNMethodContext. Object not of type \"Map\"");
        }

        @SuppressWarnings("unchecked")
        Map<String, Object> methodMap = (Map<String, Object>) in;

        KNNEngine engine = KNNEngine.DEFAULT;
        SpaceType spaceType = SpaceType.DEFAULT;
        String name = "";
        Map<String, Object> parameters = null;

        String key;
        Object value;
        for (Map.Entry<String, Object> methodEntry : methodMap.entrySet()) {
            key = methodEntry.getKey();
            value = methodEntry.getValue();
            if (KNN_ENGINE.equals(key)) {
                if (value != null && !(value instanceof String)) {
                    throw new MapperParsingException("\"" + KNN_ENGINE + "\" must be a string");
                }

                if (value != null) {
                    try {
                        engine = KNNEngine.getEngine((String) value);
                    } catch (IllegalArgumentException iae) {
                        throw new MapperParsingException("Invalid " + KNN_ENGINE + ": " + value);
                    }
                }
            } else if (SPACE_TYPE.equals(key)) {
                if (value != null && !(value instanceof String)) {
                    throw new MapperParsingException("\"" + SPACE_TYPE + "\" must be a string");
                }

                try {
                    spaceType = SpaceType.getSpace((String) value);
                } catch (IllegalArgumentException iae) {
                    throw new MapperParsingException("Invalid " + SPACE_TYPE + ": " + value);
                }
            } else if (NAME.equals(key)) {
                if (!(value instanceof String)) {
                    throw new MapperParsingException(NAME + "has to be a string");
                }

                name = (String) value;
            } else if (PARAMETERS.equals(key)) {
                if (value != null && !(value instanceof Map)) {
                    throw new MapperParsingException("Unable to parse parameters for main method component");
                }

                @SuppressWarnings("unchecked")
                Map<String, Object> parameters1 = (Map<String, Object>) value;
                parameters = parameters1;
            } else {
                throw new MapperParsingException("Invalid parameter: " + key);
            }
        }

        if (name.isEmpty()) {
            throw new MapperParsingException(NAME + " needs to be set");
        }

        MethodComponentContext method = new MethodComponentContext(name, parameters);

        return new KNNMethodContext(engine, spaceType, method);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.field(KNN_ENGINE, knnEngine.getName());
        builder.field(SPACE_TYPE, spaceType.getValue());
        builder = methodComponent.toXContent(builder, params);
        return builder;
    }
}

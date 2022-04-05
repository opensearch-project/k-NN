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

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.common.ValidationException;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.io.stream.Writeable;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.common.xcontent.ToXContentFragment;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.index.mapper.MapperParsingException;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;
import org.apache.commons.lang.builder.EqualsBuilder;
import org.apache.commons.lang.builder.HashCodeBuilder;

import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

/**
 * KNNMethodContext will contain the information necessary to produce a library index from an Opensearch mapping.
 * It will encompass all parameters necessary to build the index.
 */
public class KNNMethodContext implements ToXContentFragment, Writeable {

    private static final Logger logger = LogManager.getLogger(KNNMethodContext.class);

    private static KNNMethodContext defaultInstance = null;

    public static synchronized KNNMethodContext getDefault() {
        if (defaultInstance == null) {
            defaultInstance = new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.DEFAULT,
                    new MethodComponentContext(METHOD_HNSW, Collections.emptyMap()));
        }
        return defaultInstance;
    }

    private final KNNEngine knnEngine;
    private final SpaceType spaceType;
    private final MethodComponentContext methodComponent;

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
     * Constructor from stream.
     *
     * @param in StreamInput
     * @throws IOException on stream failure
     */
    public KNNMethodContext(StreamInput in) throws IOException {
        this.knnEngine = KNNEngine.getEngine(in.readString());
        this.spaceType = SpaceType.getSpace(in.readString());
        this.methodComponent = new MethodComponentContext(in);
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
     * @return ValidationException produced by validation errors; null if no validations errors.
     */
    public ValidationException validate() {
        return knnEngine.validateMethod(this);
    }

    /**
     * This method returns whether training is requires or not from knnEngine
     *
     * @return true if training is required by knnEngine; false otherwise
     */
    public boolean isTrainingRequired() {
        return knnEngine.isTrainingRequired(this);
    }

    /**
     * This method estimates the overhead the knn method adds irrespective of the number of vectors
     *
     * @param dimension dimension to make estimate with
     * @return size in Kilobytes
     */
    public int estimateOverheadInKB(int dimension) {
        return knnEngine.estimateOverheadInKB(this, dimension);
    }

    /**
     * Parses an Object into a KNNMethodContext.
     *
     * @param in Object containing mapping to be parsed
     * @return KNNMethodContext
     */
    public static KNNMethodContext parse(Object in) {
        if (!(in instanceof Map<?, ?>)) {
            throw new MapperParsingException("Unable to parse mapping into KNNMethodContext. Object not of type \"Map\"");
        }

        @SuppressWarnings("unchecked")
        Map<String, Object> methodMap = (Map<String, Object>) in;

        KNNEngine engine = KNNEngine.DEFAULT; // Get or default
        SpaceType spaceType = SpaceType.DEFAULT; // Get or default
        String name = "";
        Map<String, Object> parameters = new HashMap<>();

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
            } else if (METHOD_PARAMETER_SPACE_TYPE.equals(key)) {
                if (value != null && !(value instanceof String)) {
                    throw new MapperParsingException("\"" + METHOD_PARAMETER_SPACE_TYPE + "\" must be a string");
                }

                try {
                    spaceType = SpaceType.getSpace((String) value);
                } catch (IllegalArgumentException iae) {
                    throw new MapperParsingException("Invalid " + METHOD_PARAMETER_SPACE_TYPE + ": " + value);
                }
            } else if (NAME.equals(key)) {
                if (!(value instanceof String)) {
                    throw new MapperParsingException(NAME + "has to be a string");
                }

                name = (String) value;
            } else if (PARAMETERS.equals(key)) {
                if (value == null) {
                    parameters = null;
                    continue;
                }

                if (!(value instanceof Map)) {
                    throw new MapperParsingException("Unable to parse parameters for main method component");
                }

                // Interpret all map parameters as sub-MethodComponentContexts
                @SuppressWarnings("unchecked")
                Map<String, Object> parameters1 = ((Map<String, Object>) value).entrySet().stream().collect(Collectors.toMap(
                        Map.Entry::getKey, e -> {
                            Object v = e.getValue();
                            if (v instanceof Map) {
                                return MethodComponentContext.parse(v);
                            }
                            return v;
                        }
                ));

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
        builder.field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue());
        builder = methodComponent.toXContent(builder, params);
        return builder;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null || getClass() != obj.getClass())
            return false;
        KNNMethodContext other = (KNNMethodContext) obj;

        EqualsBuilder equalsBuilder = new EqualsBuilder();
        equalsBuilder.append(knnEngine, other.knnEngine);
        equalsBuilder.append(spaceType, other.spaceType);
        equalsBuilder.append(methodComponent, other.methodComponent);

        return equalsBuilder.isEquals();
    }

    @Override
    public int hashCode() {
        return new HashCodeBuilder().append(knnEngine).append(spaceType).append(methodComponent).toHashCode();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeString(knnEngine.getName());
        out.writeString(spaceType.getValue());
        this.methodComponent.writeTo(out);
    }
}

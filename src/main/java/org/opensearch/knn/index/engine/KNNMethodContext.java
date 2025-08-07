/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.opensearch.common.ValidationException;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.common.io.stream.Writeable;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.core.xcontent.ToXContentFragment;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.mapper.MapperParsingException;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;
import org.apache.commons.lang.builder.EqualsBuilder;
import org.apache.commons.lang.builder.HashCodeBuilder;

import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

/**
 * KNNMethodContext will contain the information necessary to produce a library index from an Opensearch mapping.
 * It will encompass all parameters necessary to build the index.
 */
@AllArgsConstructor(access = AccessLevel.PACKAGE)
@Getter
public class KNNMethodContext implements ToXContentFragment, Writeable {

    @NonNull
    private KNNEngine knnEngine;
    @NonNull
    @Setter
    private SpaceType spaceType;
    @NonNull
    private final MethodComponentContext methodComponentContext;
    // Currently, the KNNEngine member variable cannot be null and defaults during parsing to nmslib. However, in order
    // to support disk based engine resolution, this value potentially needs to be updated. Thus, this value is used
    // to determine if the variable can be overridden or not based on whether the user explicitly set the value during parsing
    private boolean isEngineConfigured;

    /**
     * Copy constructor. Useful for creating a deep copy of a {@link KNNMethodContext}. Note that the engine and
     * space type should be set.
     *
     * @param knnMethodContext original {@link KNNMethodContext}. Must NOT be null
     */
    public KNNMethodContext(KNNMethodContext knnMethodContext) {
        if (knnMethodContext == null) {
            throw new IllegalArgumentException("KNNMethodContext cannot be null");
        }

        this.knnEngine = knnMethodContext.knnEngine;
        this.spaceType = knnMethodContext.spaceType;
        this.isEngineConfigured = true;
        this.methodComponentContext = new MethodComponentContext(knnMethodContext.methodComponentContext);
    }

    /**
     *
     * @param knnEngine {@link KNNEngine}
     * @param spaceType {@link SpaceType}
     * @param methodComponentContext {@link MethodComponentContext}
     */
    public KNNMethodContext(KNNEngine knnEngine, SpaceType spaceType, MethodComponentContext methodComponentContext) {
        this(knnEngine, spaceType, methodComponentContext, true);
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
        this.methodComponentContext = new MethodComponentContext(in);
        this.isEngineConfigured = true;
    }

    /**
     * Set the {@link KNNEngine} if it is not configured (i.e. DEFAULT). This is useful for using different engines
     * for different configurations - i.e. dynamic defaults
     *
     * @param knnEngine KNNEngine to set
     */
    public void setKnnEngine(KNNEngine knnEngine) {
        if (isEngineConfigured) {
            throw new IllegalArgumentException("Cannot configure KNNEngine if it has already been configured");
        }
        this.knnEngine = knnEngine;
        this.isEngineConfigured = true;
    }

    /**
     * This method uses the knnEngine to validate that the method is compatible with the engine.
     *
     * @param knnMethodConfigContext context to validate against
     * @return ValidationException produced by validation errors; null if no validations errors.
     */
    public ValidationException validate(KNNMethodConfigContext knnMethodConfigContext) {
        return knnEngine.validateMethod(this, knnMethodConfigContext);
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
     * @param knnMethodConfigContext context to estimate overhead
     * @return size in Kilobytes
     */
    public int estimateOverheadInKB(KNNMethodConfigContext knnMethodConfigContext) {
        return knnEngine.estimateOverheadInKB(this, knnMethodConfigContext);
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

        boolean isEngineConfigured = false;
        KNNEngine engine = KNNEngine.UNDEFINED; // Get or default
        SpaceType spaceType = SpaceType.UNDEFINED; // Get or default
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
                isEngineConfigured = true;
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
                Map<String, Object> parameters1 = ((Map<String, Object>) value).entrySet()
                    .stream()
                    .collect(Collectors.toMap(Map.Entry::getKey, e -> {
                        Object v = e.getValue();
                        if (v instanceof Map) {
                            return MethodComponentContext.parse(v);
                        }
                        return v;
                    }));

                parameters = parameters1;
            } else {
                throw new MapperParsingException("Invalid parameter: " + key);
            }
        }

        if (name.isEmpty()) {
            throw new MapperParsingException(NAME + " needs to be set");
        }

        MethodComponentContext method = new MethodComponentContext(name, parameters);

        return new KNNMethodContext(engine, spaceType, method, isEngineConfigured);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.field(KNN_ENGINE, knnEngine.getName());
        builder.field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue());
        builder = methodComponentContext.toXContent(builder, params);
        return builder;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        KNNMethodContext other = (KNNMethodContext) obj;

        EqualsBuilder equalsBuilder = new EqualsBuilder();
        equalsBuilder.append(knnEngine, other.knnEngine);
        equalsBuilder.append(spaceType, other.spaceType);
        equalsBuilder.append(methodComponentContext, other.methodComponentContext);

        return equalsBuilder.isEquals();
    }

    @Override
    public int hashCode() {
        return new HashCodeBuilder().append(knnEngine).append(spaceType).append(methodComponentContext).toHashCode();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeString(knnEngine.getName());
        out.writeString(spaceType.getValue());
        this.methodComponentContext.writeTo(out);
    }
}

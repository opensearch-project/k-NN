/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NonNull;
import org.opensearch.Version;
import org.opensearch.common.Nullable;
import org.opensearch.core.common.Strings;
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
import java.util.Optional;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.NMSLIB_NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

/**
 * KNNMethodContext will contain the information necessary to produce a library index from an Opensearch mapping.
 * It will encompass all parameters necessary to build the index.
 */
@AllArgsConstructor
@EqualsAndHashCode
public class KNNMethodContext implements ToXContentFragment, Writeable {
    private static final String UNDEFINED_VALUE = "undefined";

    private static final StreamHelper DEFAULT_STREAM_HELPER = new DefaultStreamHelper();
    private static final StreamHelper BEFORE_217_STREAM_HELPER = new Before217StreamHelper();

    @Nullable
    private final KNNEngine knnEngine;
    @Nullable
    private final SpaceType spaceType;
    @NonNull
    @Getter
    private final MethodComponentContext methodComponentContext;

    /**
     * Constructor from stream.
     *
     * @param in StreamInput
     * @throws IOException on stream failure
     */
    public KNNMethodContext(StreamInput in) throws IOException {
        StreamHelper streamHelper = in.getVersion().onOrAfter(Version.V_2_17_0) ? DEFAULT_STREAM_HELPER : BEFORE_217_STREAM_HELPER;
        this.knnEngine = streamHelper.streamInKNNEngine(in);
        this.spaceType = streamHelper.streamInSpaceType(in);
        this.methodComponentContext = streamHelper.streamInMethodComponentContext(in);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        StreamHelper streamHelper = out.getVersion().onOrAfter(Version.V_2_17_0) ? DEFAULT_STREAM_HELPER : BEFORE_217_STREAM_HELPER;
        streamHelper.streamOutKNNEngine(out, knnEngine);
        streamHelper.streamOutSpaceType(out, spaceType);
        streamHelper.streamOutMethodComponentContext(out, methodComponentContext);
    }

    /**
     * Get the KNN Engine
     *
     * @return KNNEngine
     */
    public Optional<KNNEngine> getKnnEngine() {
        return Optional.ofNullable(knnEngine);
    }

    /**
     * Get the Space Type
     *
     * @return SpaceType
     */
    public Optional<SpaceType> getSpaceType() {
        return Optional.ofNullable(spaceType);
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

        KNNEngine engine = null;
        SpaceType spaceType = null;
        String name = null;
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

        MethodComponentContext method = new MethodComponentContext(name, parameters);

        return new KNNMethodContext(engine, spaceType, method);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        if (knnEngine != null) {
            builder.field(KNN_ENGINE, knnEngine.getName());
        }

        if (spaceType != null) {
            builder.field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue());
        }
        return methodComponentContext.toXContent(builder, params);
    }

    private interface StreamHelper {
        KNNEngine streamInKNNEngine(StreamInput in) throws IOException;

        void streamOutKNNEngine(StreamOutput out, KNNEngine value) throws IOException;

        SpaceType streamInSpaceType(StreamInput in) throws IOException;

        void streamOutSpaceType(StreamOutput out, SpaceType value) throws IOException;

        MethodComponentContext streamInMethodComponentContext(StreamInput in) throws IOException;

        void streamOutMethodComponentContext(StreamOutput out, MethodComponentContext value) throws IOException;
    }

    private static class DefaultStreamHelper implements StreamHelper {
        @Override
        public KNNEngine streamInKNNEngine(StreamInput in) throws IOException {
            String knnEngineString = in.readOptionalString();
            return knnEngineString != null ? KNNEngine.getEngine(knnEngineString) : null;
        }

        @Override
        public void streamOutKNNEngine(StreamOutput out, KNNEngine value) throws IOException {
            String knnEngineString = value != null ? value.getName() : null;
            out.writeOptionalString(knnEngineString);
        }

        @Override
        public SpaceType streamInSpaceType(StreamInput in) throws IOException {
            String spaceTypeString = in.readOptionalString();
            return spaceTypeString != null ? SpaceType.getSpace(spaceTypeString) : null;
        }

        @Override
        public void streamOutSpaceType(StreamOutput out, SpaceType value) throws IOException {
            String spaceTypeString = value != null ? value.getValue() : null;
            out.writeOptionalString(spaceTypeString);
        }

        @Override
        public MethodComponentContext streamInMethodComponentContext(StreamInput in) throws IOException {
            return new MethodComponentContext(in);
        }

        @Override
        public void streamOutMethodComponentContext(StreamOutput out, MethodComponentContext value) throws IOException {
            value.writeTo(out);
        }
    }

    private static class Before217StreamHelper implements StreamHelper {
        @Override
        public KNNEngine streamInKNNEngine(StreamInput in) throws IOException {
            return KNNEngine.getEngine(in.readString());
        }

        @Override
        public void streamOutKNNEngine(StreamOutput out, KNNEngine value) throws IOException {
            // This may happen in a mixed cluster state. If this is the case, we need to write the default engine
            if (value == null) {
                out.writeString(NMSLIB_NAME);
            } else {
                out.writeString(value.getName());
            }
        }

        @Override
        public SpaceType streamInSpaceType(StreamInput in) throws IOException {
            String spaceTypeString = in.readString();
            if (Strings.isEmpty(spaceTypeString) || UNDEFINED_VALUE.equals(spaceTypeString)) {
                return null;
            }
            return SpaceType.getSpace(spaceTypeString);
        }

        @Override
        public void streamOutSpaceType(StreamOutput out, SpaceType value) throws IOException {
            if (value == null) {
                out.writeString(UNDEFINED_VALUE);
            } else {
                out.writeString(value.getValue());
            }
        }

        @Override
        public MethodComponentContext streamInMethodComponentContext(StreamInput in) throws IOException {
            return new MethodComponentContext(in);
        }

        @Override
        public void streamOutMethodComponentContext(StreamOutput out, MethodComponentContext value) throws IOException {
            value.writeTo(out);
        }
    }
}

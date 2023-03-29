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

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.io.stream.Writeable;
import org.opensearch.core.xcontent.ToXContentFragment;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.mapper.MapperParsingException;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;
import org.apache.commons.lang.builder.EqualsBuilder;
import org.apache.commons.lang.builder.HashCodeBuilder;

import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

/**
 * MethodComponentContext represents a single user provided building block of a knn library index.
 *
 * Each component is composed of a name and a map of parameters.
 */
@AllArgsConstructor
public class MethodComponentContext implements ToXContentFragment, Writeable {

    @Getter
    private final String name;
    private final Map<String, Object> parameters;

    /**
     * Constructor from stream.
     *
     * @param in StreamInput
     * @throws IOException on stream failure
     */
    public MethodComponentContext(StreamInput in) throws IOException {
        this.name = in.readString();

        // Due to backwards compatibility issue, parameters could be null. To prevent any null pointer exceptions,
        // do not read if their are no bytes left is null. Make sure this is in sync with the fellow read method. For
        // more information, refer to https://github.com/opensearch-project/k-NN/issues/353.
        if (in.available() > 0) {
            this.parameters = in.readMap(StreamInput::readString, new ParameterMapValueReader());
        } else {
            this.parameters = null;
        }
    }

    /**
     * Parses the object into MethodComponentContext
     *
     * @param in Object to be parsed
     * @return MethodComponentContext
     */
    public static MethodComponentContext parse(Object in) {
        if (!(in instanceof Map<?, ?>)) {
            throw new MapperParsingException("Unable to parse MethodComponent");
        }

        @SuppressWarnings("unchecked")
        Map<String, Object> methodMap = (Map<String, Object>) in;
        String name = "";
        Map<String, Object> parameters = new HashMap<>();

        String key;
        Object value;

        for (Map.Entry<String, Object> methodEntry : methodMap.entrySet()) {
            key = methodEntry.getKey();
            value = methodEntry.getValue();

            if (NAME.equals(key)) {
                if (!(value instanceof String)) {
                    throw new MapperParsingException("Component name should be a string");
                }
                name = (String) value;
            } else if (PARAMETERS.equals(key)) {
                if (value == null) {
                    parameters = null;
                    continue;
                }

                if (!(value instanceof Map)) {
                    throw new MapperParsingException("Unable to parse parameters for  method component");
                }

                // Check to interpret map parameters as sub-methodComponentContexts
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
                throw new MapperParsingException("Invalid parameter for MethodComponentContext: " + key);
            }
        }

        if (name.isEmpty()) {
            throw new MapperParsingException(NAME + " needs to be set");
        }

        return new MethodComponentContext(name, parameters);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.field(NAME, name);
        // Due to backwards compatibility issue, parameters could be null. To prevent any null pointer exceptions,
        // we just create the null field. If parameters are not null, we created a nested structure. For more
        // information, refer to https://github.com/opensearch-project/k-NN/issues/353.
        if (parameters == null) {
            builder.field(PARAMETERS, (String) null);
        } else {
            builder.startObject(PARAMETERS);
            parameters.forEach((key, value) -> {
                try {
                    if (value instanceof MethodComponentContext) {
                        builder.startObject(key);
                        ((MethodComponentContext) value).toXContent(builder, params);
                        builder.endObject();
                    } else {
                        builder.field(key, value);
                    }
                } catch (IOException ioe) {
                    throw new RuntimeException("Unable to generate xcontent for method component");
                }

            });
            builder.endObject();
        }

        return builder;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        MethodComponentContext other = (MethodComponentContext) obj;

        EqualsBuilder equalsBuilder = new EqualsBuilder();
        equalsBuilder.append(name, other.name);
        equalsBuilder.append(parameters, other.parameters);
        return equalsBuilder.isEquals();
    }

    @Override
    public int hashCode() {
        return new HashCodeBuilder().append(name).append(parameters).toHashCode();
    }

    /**
     * Gets the parameters of the component
     *
     * @return parameters
     */
    public Map<String, Object> getParameters() {
        // Due to backwards compatibility issue, parameters could be null. To prevent any null pointer exceptions,
        // return an empty map if parameters is null. For more information, refer to
        // https://github.com/opensearch-project/k-NN/issues/353.
        if (parameters == null) {
            return Collections.emptyMap();
        }
        return parameters;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeString(this.name);

        // Due to backwards compatibility issue, parameters could be null. To prevent any null pointer exceptions,
        // do not write if parameters is null. Make sure this is in sync with the fellow read method. For more
        // information, refer to https://github.com/opensearch-project/k-NN/issues/353.
        if (this.parameters != null) {
            out.writeMap(this.parameters, StreamOutput::writeString, new ParameterMapValueWriter());
        }
    }

    // Because the generic StreamOutput writeMap method can only write generic values, we need to create a custom one
    // that handles the case when a parameter value is another method component context.
    private static class ParameterMapValueWriter implements Writer<Object> {

        private ParameterMapValueWriter() {}

        @Override
        public void write(StreamOutput out, Object o) throws IOException {
            if (o instanceof MethodComponentContext) {
                out.writeBoolean(true);
                ((MethodComponentContext) o).writeTo(out);
            } else {
                out.writeBoolean(false);
                out.writeGenericValue(o);
            }
        }
    }

    // Because the generic StreamInput writeMap method can only read generic values, we need to create a custom one
    // that handles the case when a parameter value is another method component context.
    private static class ParameterMapValueReader implements Reader<Object> {

        private ParameterMapValueReader() {}

        @Override
        public Object read(StreamInput in) throws IOException {
            boolean isValueMethodComponentContext = in.readBoolean();
            if (isValueMethodComponentContext) {
                return new MethodComponentContext(in);
            }
            return in.readGenericValue();
        }
    }
}

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

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import org.apache.commons.lang.math.NumberUtils;
import org.opensearch.Version;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.common.io.stream.Writeable;
import org.opensearch.core.xcontent.ToXContentFragment;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.mapper.MapperParsingException;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;
import org.apache.commons.lang.builder.EqualsBuilder;
import org.apache.commons.lang.builder.HashCodeBuilder;
import org.opensearch.knn.indices.ModelMetadata;

import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

/**
 * MethodComponentContext represents a single user provided building block of a knn library index.
 *
 * Each component is composed of a name and a map of parameters.
 */
@RequiredArgsConstructor
public class MethodComponentContext implements ToXContentFragment, Writeable {

    public static final MethodComponentContext DEFAULT = new MethodComponentContext("", Collections.emptyMap());

    private static final String DELIMITER = ";";

    @Getter
    private final String name;
    private final Map<String, Object> parameters;

    @Getter
    @Setter
    private Version indexVersion;

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

    /**
     *
     * Provides a String representation of MethodComponentContext
     * Sample return:
     * {name=ivf;parameters=[nlist=4;type=fp16;encoder={name=sq;parameters=[nprobes=2;clip=false;]};]}
     *
     * @return string representation
     */
    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("{name=").append(name).append(DELIMITER);
        stringBuilder.append("parameters=[");
        if (Objects.nonNull(parameters)) {
            for (Map.Entry<String, Object> entry : parameters.entrySet()) {
                stringBuilder.append(entry.getKey()).append("=");
                String value = entry.getValue().toString();
                // Model Metadata uses a delimiter to split the input string in its fromString method
                // If any of the values in the method component context contain this delimiter,
                // then the method will not work correctly. Therefore, we replace the delimiter with an uncommon
                // sequence that is very unlikely to appear in the value itself.
                value = value.replace(ModelMetadata.DELIMITER, "$%$");
                stringBuilder.append(value).append(DELIMITER);
            }
        }
        stringBuilder.append("]}");
        return stringBuilder.toString();
    }

    /**
     * This method converts a string created by the toString() method of MethodComponentContext
     * to a MethodComponentContext object.
     *
     * @param in a string representation of MethodComponentContext
     * @return a MethodComponentContext object
     */
    public static MethodComponentContext fromString(String in) {
        int index = 0;
        String[] outerMethodComponentContextArray = in.split("\\{", -1);
        if (outerMethodComponentContextArray[index].isEmpty()) {
            index++;
        }
        String[] innerMethodComponentContextArray = outerMethodComponentContextArray[index].split(DELIMITER, -1);
        index++;
        String name = "";
        name = innerMethodComponentContextArray[0].substring(innerMethodComponentContextArray[0].indexOf("=") + 1);
        Map<String, Object> parameters = parseParameters(innerMethodComponentContextArray, outerMethodComponentContextArray, index);

        return new MethodComponentContext(name, parameters);
    }

    private static Map<String,Object> parseParameters(String[] innerMethodComponentContextArray, String[] outerMethodComponentContextArray, int index) {
        Map<String, Object> parameters = new HashMap<>();
        if (innerMethodComponentContextArray.length > 2) {
            for (int i = 1; i < innerMethodComponentContextArray.length; i++) {
                String substring = innerMethodComponentContextArray[i];
                if (i == 1) {
                    substring = substring.substring(substring.indexOf("=") + 2);
                }
                if (substring.charAt(0) == ']') {
                    break;
                }
                String key = substring.substring(0, substring.indexOf("="));
                String stringValue = substring.substring(substring.indexOf("=") + 1);
                Object value;
                // Parameters will always be a MethodComponentContext, String, integer, or boolean
                // https://github.com/opensearch-project/k-NN/blob/2.12/src/main/java/org/opensearch/knn/index/Parameter.java
                if (stringValue.isEmpty()) {
                    value = fromString(outerMethodComponentContextArray[index]);
                } else if (NumberUtils.isNumber(stringValue)) {
                    value = Integer.parseInt(stringValue);
                } else if (stringValue.equals("true") || stringValue.equals("false")) {
                    value = Boolean.parseBoolean(stringValue);
                } else {
                    stringValue = stringValue.replace("$%$", ModelMetadata.DELIMITER);
                    value = stringValue;
                }

                parameters.put(key, value);
            }
        } else {
            parameters = Collections.emptyMap();
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

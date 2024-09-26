/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.apache.commons.lang.math.NumberUtils;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.common.io.stream.Writeable;
import org.opensearch.core.xcontent.ToXContentFragment;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
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

    // EMPTY method component context can only occur if a model originated on a cluster before 2.13.0 and the cluster is then upgraded to
    // 2.13.0
    public static final MethodComponentContext EMPTY = new MethodComponentContext("", Collections.emptyMap());

    private static final String DELIMITER = ";";
    private static final String DELIMITER_PLACEHOLDER = "$%$";

    @Getter
    private final String name;
    private final Map<String, Object> parameters;

    /**
     * Copy constructor. Creates a deep copy of a {@link MethodComponentContext}
     *
     * @param methodComponentContext to be copied. Must NOT be null
     */
    public MethodComponentContext(MethodComponentContext methodComponentContext) {
        if (methodComponentContext == null) {
            throw new IllegalArgumentException("MethodComponentContext cannot be null");
        }

        this.name = methodComponentContext.name;
        this.parameters = new HashMap<>();
        if (methodComponentContext.parameters != null) {
            for (Map.Entry<String, Object> entry : methodComponentContext.parameters.entrySet()) {
                if (entry.getValue() instanceof MethodComponentContext) {
                    parameters.put(entry.getKey(), new MethodComponentContext((MethodComponentContext) entry.getValue()));
                } else {
                    parameters.put(entry.getKey(), entry.getValue());
                }
            }
        }
    }

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

    public static MethodComponentContext fromXContent(XContentParser xContentParser) throws IOException {
        // If it is a fresh parser, move to the first token
        if (xContentParser.currentToken() == null) {
            xContentParser.nextToken();
        }
        Map<String, Object> parsedMap = xContentParser.map();
        return MethodComponentContext.parse(parsedMap);
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
    public String toClusterStateString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("{name=").append(name).append(DELIMITER);
        stringBuilder.append("parameters=[");
        if (Objects.nonNull(parameters)) {
            for (Map.Entry<String, Object> entry : parameters.entrySet()) {
                stringBuilder.append(entry.getKey()).append("=");
                Object objectValue = entry.getValue();
                String value;
                if (objectValue instanceof MethodComponentContext) {
                    value = ((MethodComponentContext) objectValue).toClusterStateString();
                } else {
                    value = entry.getValue().toString();
                }
                // Model Metadata uses a delimiter to split the input string in its fromString method
                // https://github.com/opensearch-project/k-NN/blob/2.12/src/main/java/org/opensearch/knn/indices/ModelMetadata.java#L265
                // If any of the values in the method component context contain this delimiter,
                // then the method will not work correctly. Therefore, we replace the delimiter with an uncommon
                // sequence that is very unlikely to appear in the value itself.
                // https://github.com/opensearch-project/k-NN/issues/1337
                value = value.replace(ModelMetadata.DELIMITER, DELIMITER_PLACEHOLDER);
                stringBuilder.append(value).append(DELIMITER);
            }
        }
        stringBuilder.append("]}");
        return stringBuilder.toString();
    }

    /**
     * This method converts a string created by the toClusterStateString() method of MethodComponentContext
     * to a MethodComponentContext object.
     *
     * @param in a string representation of MethodComponentContext
     * @return a MethodComponentContext object
     */
    public static MethodComponentContext fromClusterStateString(String in) {
        String stringToParse = unwrapString(in, '{', '}');

        // Parse name from string
        String[] nameAndParameters = stringToParse.split(DELIMITER, 2);
        checkExpectedArrayLength(nameAndParameters, 2);
        String name = parseName(nameAndParameters[0]);
        String parametersString = nameAndParameters[1];
        Map<String, Object> parameters = parseParameters(parametersString);
        return new MethodComponentContext(name, parameters);
    }

    private static String parseName(String candidateNameString) {
        // Expecting candidateNameString to look like "name=ivf"
        checkStringNotEmpty(candidateNameString);
        String[] nameKeyAndValue = candidateNameString.split("=");
        checkStringMatches(nameKeyAndValue[0], "name");
        if (nameKeyAndValue.length == 1) {
            return "";
        }
        checkExpectedArrayLength(nameKeyAndValue, 2);
        return nameKeyAndValue[1];
    }

    private static Map<String, Object> parseParameters(String candidateParameterString) {
        checkStringNotEmpty(candidateParameterString);
        String[] parametersKeyAndValue = candidateParameterString.split("=", 2);
        checkStringMatches(parametersKeyAndValue[0], "parameters");
        if (parametersKeyAndValue.length == 1) {
            return Collections.emptyMap();
        }
        checkExpectedArrayLength(parametersKeyAndValue, 2);
        return parseParametersValue(parametersKeyAndValue[1]);
    }

    private static Map<String, Object> parseParametersValue(String candidateParameterValueString) {
        // Expected input is [nlist=4;type=fp16;encoder={name=sq;parameters=[nprobes=2;clip=false;]};]
        checkStringNotEmpty(candidateParameterValueString);
        candidateParameterValueString = unwrapString(candidateParameterValueString, '[', ']');
        Map<String, Object> parameters = new HashMap<>();
        while (!candidateParameterValueString.isEmpty()) {
            String[] keyAndValueToParse = candidateParameterValueString.split("=", 2);
            if (keyAndValueToParse.length == 1 && keyAndValueToParse[0].charAt(0) == ';') {
                break;
            }
            String key = keyAndValueToParse[0];
            ValueAndRestToParse parsed = parseParameterValueAndRestToParse(keyAndValueToParse[1]);
            parameters.put(key, parsed.getValue());
            candidateParameterValueString = parsed.getRestToParse();
        }

        return parameters;
    }

    private static ValueAndRestToParse parseParameterValueAndRestToParse(String candidateParameterValueAndRestToParse) {
        if (candidateParameterValueAndRestToParse.charAt(0) == '{') {
            int endOfNestedMap = findClosingPosition(candidateParameterValueAndRestToParse, '{', '}');
            String nestedMethodContext = candidateParameterValueAndRestToParse.substring(0, endOfNestedMap + 1);
            Object nestedParse = fromClusterStateString(nestedMethodContext);
            String restToParse = candidateParameterValueAndRestToParse.substring(endOfNestedMap + 1);
            return new ValueAndRestToParse(nestedParse, restToParse);
        }

        String[] stringValueAndRestToParse = candidateParameterValueAndRestToParse.split(DELIMITER, 2);
        String stringValue = stringValueAndRestToParse[0];
        Object value;
        if (NumberUtils.isNumber(stringValue)) {
            value = Integer.parseInt(stringValue);
        } else if (stringValue.equals("true") || stringValue.equals("false")) {
            value = Boolean.parseBoolean(stringValue);
        } else {
            stringValue = stringValue.replace(DELIMITER_PLACEHOLDER, ModelMetadata.DELIMITER);
            value = stringValue;
        }

        return new ValueAndRestToParse(value, stringValueAndRestToParse[1]);
    }

    private static String unwrapString(String in, char expectedStart, char expectedEnd) {
        if (in.length() < 2) {
            throw new IllegalArgumentException("Invalid string.");
        }

        if (in.charAt(0) != expectedStart || in.charAt(in.length() - 1) != expectedEnd) {
            throw new IllegalArgumentException("Invalid string." + in);
        }
        return in.substring(1, in.length() - 1);
    }

    private static int findClosingPosition(String in, char expectedStart, char expectedEnd) {
        int nestedLevel = 0;
        for (int i = 0; i < in.length(); i++) {
            if (in.charAt(i) == expectedStart) {
                nestedLevel++;
                continue;
            }

            if (in.charAt(i) == expectedEnd) {
                nestedLevel--;
            }

            if (nestedLevel == 0) {
                return i;
            }
        }

        throw new IllegalArgumentException("Invalid string. No end to the nesting");
    }

    private static void checkStringNotEmpty(String string) {
        if (string.isEmpty()) {
            throw new IllegalArgumentException("Unable to parse MethodComponentContext");
        }
    }

    private static void checkStringMatches(String string, String expected) {
        if (!Objects.equals(string, expected)) {
            throw new IllegalArgumentException("Unexpected key in MethodComponentContext.  Expected 'name' or 'parameters'");
        }
    }

    private static void checkExpectedArrayLength(String[] array, int expectedLength) {
        if (null == array) {
            throw new IllegalArgumentException("Error parsing MethodComponentContext.  Array is null.");
        }

        if (array.length != expectedLength) {
            throw new IllegalArgumentException("Error parsing MethodComponentContext.  Array is not expected length.");
        }
    }

    @AllArgsConstructor
    @Getter
    private static class ValueAndRestToParse {
        private final Object value;
        private final String restToParse;
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

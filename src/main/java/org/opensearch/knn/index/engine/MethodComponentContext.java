/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.apache.commons.lang.math.NumberUtils;
import org.opensearch.Version;
import org.opensearch.core.common.Strings;
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
import java.util.Optional;
import java.util.stream.Collectors;
import org.opensearch.knn.indices.ModelMetadata;

import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.index.engine.ParseUtil.checkExpectedArrayLength;
import static org.opensearch.knn.index.engine.ParseUtil.checkStringMatches;
import static org.opensearch.knn.index.engine.ParseUtil.checkStringNotEmpty;
import static org.opensearch.knn.index.engine.ParseUtil.unwrapString;

/**
 * MethodComponentContext represents a single user provided building block of a knn library index.
 *
 * Each component is composed of a name and a map of parameters.
 */
@RequiredArgsConstructor
@EqualsAndHashCode
public class MethodComponentContext implements ToXContentFragment, Writeable {

    // EMPTY method component context can only occur if a model originated on a cluster before 2.13.0 and the cluster is then upgraded to
    // 2.13.0
    public static final MethodComponentContext EMPTY = new MethodComponentContext("", Collections.emptyMap());

    private static final String DELIMITER = ";";
    private static final String DELIMITER_PLACEHOLDER = "$%$";

    private static final StreamHelper DEFAULT_STREAM_HELPER = new DefaultStreamHelper();
    private static final StreamHelper BEFORE_217_STREAM_HELPER = new Before217StreamHelper();

    private final String name;
    private final Map<String, Object> parameters;

    /**
     * Constructor from stream.
     *
     * @param in StreamInput
     * @throws IOException on stream failure
     */
    public MethodComponentContext(StreamInput in) throws IOException {
        StreamHelper streamHelper = in.getVersion().onOrAfter(Version.V_2_17_0) ? DEFAULT_STREAM_HELPER : BEFORE_217_STREAM_HELPER;
        this.name = streamHelper.streamInName(in);
        this.parameters = streamHelper.streamInParameters(in);
    }

    /**
     * Get name of the method component context
     *
     * @return Get name
     */
    public Optional<String> getName() {
        return Optional.ofNullable(name);
    }

    /**
     * Get parameters of the method component context
     *
     * @return Parameters
     */
    public Optional<Map<String, Object>> getParameters() {
        return Optional.ofNullable(parameters);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        StreamHelper streamHelper = out.getVersion().onOrAfter(Version.V_2_17_0) ? DEFAULT_STREAM_HELPER : BEFORE_217_STREAM_HELPER;
        streamHelper.streamOutName(out, name);
        streamHelper.streamOutParameters(out, parameters);
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
        String name = null;
        Map<String, Object> parameters = null;

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
                parameters = ((Map<?, ?>) value).entrySet().stream().collect(Collectors.toMap(v -> {
                    if (v.getKey() instanceof String) {
                        return (String) v.getKey();
                    }
                    throw new MapperParsingException("Invalid type for input map for MethodComponentContext");
                }, e -> {
                    Object v = e.getValue();
                    if (v instanceof Map) {
                        return MethodComponentContext.parse(v);
                    }
                    return v;
                }));
            } else {
                throw new MapperParsingException("Invalid parameter for MethodComponentContext: " + key);
            }
        }

        return new MethodComponentContext(name, parameters);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        if (name != null) {
            builder.field(NAME, name);
        }

        // Due to backwards compatibility issue, parameters could be null. To prevent any null pointer exceptions,
        // we just create the null field. If parameters are not null, we created a nested structure. For more
        // information, refer to https://github.com/opensearch-project/k-NN/issues/353.
        if (parameters != null) {
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
        stringBuilder.append("{");
        boolean isNameNull = true;
        if (name != null) {
            stringBuilder.append("name=").append(name);
            isNameNull = false;
        }

        if (parameters != null) {
            if (!isNameNull) {
                stringBuilder.append(DELIMITER);
            }
            stringBuilder.append("parameters=[");
            parametersToClusterStateString(stringBuilder);
            stringBuilder.append("]");
        }
        stringBuilder.append("}");
        return stringBuilder.toString();
    }

    private void parametersToClusterStateString(StringBuilder stringBuilder) {
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

    /**
     * This method converts a string created by the toClusterStateString() method of MethodComponentContext
     * to a MethodComponentContext object.
     *
     * @param in a string representation of MethodComponentContext
     * @return a MethodComponentContext object
     */
    public static MethodComponentContext fromClusterStateString(String in) {
        String stringToParse = unwrapString(in, '{', '}');
        String name = null;
        Map<String, Object> parameters = null;
        if (Strings.isEmpty(stringToParse)) {
            return new MethodComponentContext(name, parameters);
        }

        // Parse name from string
        String[] nameAndParameters = stringToParse.split(DELIMITER, 2);
        if (nameAndParameters.length == 1) {
            if (nameAndParameters[0].startsWith(NAME)) {
                name = parseName(nameAndParameters[0]);
            } else {
                parameters = parseParameters(nameAndParameters[0]);
            }
            return new MethodComponentContext(name, parameters);
        }

        checkExpectedArrayLength(nameAndParameters, 2);
        name = parseName(nameAndParameters[0]);
        parameters = parseParameters(nameAndParameters[1]);
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
            return null;
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
            int endOfNestedMap = ParseUtil.findClosingPosition(candidateParameterValueAndRestToParse, '{', '}');
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

    @AllArgsConstructor
    @Getter
    private static class ValueAndRestToParse {
        private final Object value;
        private final String restToParse;
    }

    private interface StreamHelper {
        String streamInName(StreamInput in) throws IOException;

        void streamOutName(StreamOutput out, String value) throws IOException;

        Map<String, Object> streamInParameters(StreamInput in) throws IOException;

        void streamOutParameters(StreamOutput out, Map<String, Object> value) throws IOException;
    }

    private static class DefaultStreamHelper implements StreamHelper {
        public String streamInName(StreamInput in) throws IOException {
            return in.readOptionalString();
        }

        public void streamOutName(StreamOutput out, String value) throws IOException {
            out.writeOptionalString(value);
        }

        public Map<String, Object> streamInParameters(StreamInput in) throws IOException {
            if (in.readBoolean() == false) {
                return null;
            }
            return in.readMap(StreamInput::readString, new ParameterMapValueReader());
        }

        public void streamOutParameters(StreamOutput out, Map<String, Object> value) throws IOException {
            if (value != null) {
                out.writeBoolean(true);
                out.writeMap(value, StreamOutput::writeString, new ParameterMapValueWriter());
            } else {
                out.writeBoolean(false);
            }
        }
    }

    // Legacy Stream helper. This logic is incorrect but works in some cases. In order to maintain compatibility with
    // older stream versions (whose code we cannot change), we need to leave this logic here.
    //
    // The relevant context for this is in https://github.com/opensearch-project/k-NN/issues/353.
    private static class Before217StreamHelper implements StreamHelper {
        public String streamInName(StreamInput in) throws IOException {
            return in.readString();
        }

        public void streamOutName(StreamOutput out, String value) throws IOException {
            out.writeString(value);
        }

        public Map<String, Object> streamInParameters(StreamInput in) throws IOException {
            if (in.available() > 0) {
                return in.readMap(StreamInput::readString, new ParameterMapValueReader());
            }
            return null;
        }

        public void streamOutParameters(StreamOutput out, Map<String, Object> value) throws IOException {
            if (value != null) {
                out.writeMap(value, StreamOutput::writeString, new ParameterMapValueWriter());
            }
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

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

package org.opensearch.knn.index.query.parser;

import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.opensearch.common.ValidationException;
import org.opensearch.core.common.ParsingException;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.index.query.request.MethodParameter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.index.IndexUtil.isClusterOnOrAfterMinRequiredVersion;
import static org.opensearch.knn.index.query.KNNQueryBuilder.METHOD_PARAMS_FIELD;
import static org.opensearch.knn.index.query.KNNQueryBuilder.NAME;
import static org.opensearch.knn.index.query.parser.KNNXParserUtil.parseJsonObject;

@EqualsAndHashCode
@Getter
@AllArgsConstructor
public class MethodParametersParser {

    // Validation on rest layer
    public static ValidationException validateMethodParameters(final Map<String, ?> methodParameters) {
        final List<String> errors = new ArrayList<>();
        for (final Map.Entry<String, ?> methodParameter : methodParameters.entrySet()) {
            final MethodParameter parameter = MethodParameter.enumOf(methodParameter.getKey());
            if (parameter != null) {
                final ValidationException validationException = parameter.validate(methodParameter.getValue());
                if (validationException != null) {
                    errors.add(validationException.getMessage());
                }
            } else { // Should never happen if used in the right sequence
                errors.add(methodParameter.getKey() + " is not a valid method parameter");
            }
        }

        if (!errors.isEmpty()) {
            ValidationException validationException = new ValidationException();
            validationException.addValidationErrors(errors);
            return validationException;
        }
        return null;
    }

    // deserialize for node to node communication
    public static Map<String, ?> streamInput(StreamInput in) throws IOException {
        if (!in.readBoolean()) {
            return null;
        }

        final Map<String, Object> methodParameters = new HashMap<>();
        for (final MethodParameter methodParameter : MethodParameter.values()) {
            if (isClusterOnOrAfterMinRequiredVersion(methodParameter.getName())) {
                String name = in.readString();
                Object value = in.readGenericValue();
                if (value != null) {
                    methodParameters.put(name, methodParameter.parse(value));
                }
            }
        }

        return !methodParameters.isEmpty() ? methodParameters : null;
    }

    // serialize for node to node communication
    public static void streamOutput(StreamOutput out, Map<String, ?> methodParameters) throws IOException {
        if (methodParameters == null || methodParameters.isEmpty()) {
            out.writeBoolean(false);
        } else {
            out.writeBoolean(true);
            // All values are written to deserialize without ambiguity
            for (final MethodParameter methodParameter : MethodParameter.values()) {
                if (isClusterOnOrAfterMinRequiredVersion(methodParameter.getName())) {
                    out.writeString(methodParameter.getName());
                    out.writeGenericValue(methodParameters.get(methodParameter.getName()));
                }
            }
        }
    }

    public static void doXContent(final XContentBuilder builder, final Map<String, ?> methodParameters) throws IOException {
        if (methodParameters == null || methodParameters.isEmpty()) {
            return;
        }
        builder.startObject(METHOD_PARAMS_FIELD.getPreferredName());
        for (final Map.Entry<String, ?> entry : methodParameters.entrySet()) {
            if (entry.getKey() != null && entry.getValue() != null) {
                builder.field(entry.getKey(), entry.getValue());
            }
        }
        builder.endObject();
    }

    public static Map<String, ?> fromXContent(final XContentParser parser) throws IOException {
        final Map<String, Object> methodParametersJson = parseJsonObject(parser);
        final Map<String, ?> methodParameters = new HashMap<>();
        for (Map.Entry<String, Object> requestParameter : methodParametersJson.entrySet()) {
            final String name = requestParameter.getKey();
            final Object value = requestParameter.getValue();
            final MethodParameter parameter = MethodParameter.enumOf(name);
            if (parameter == null) {
                throw new ParsingException(parser.getTokenLocation(), "[" + NAME + "] unknown method parameter found [" + name + "]");
            }

            try {
                // This makes sure that we throw parsing exception on rest layer.
                methodParameters.put(name, parameter.parse(value));
            } catch (final Exception exception) {
                throw new ParsingException(parser.getTokenLocation(), exception.getMessage());
            }
        }
        return methodParameters.isEmpty() ? null : methodParameters;
    }
}

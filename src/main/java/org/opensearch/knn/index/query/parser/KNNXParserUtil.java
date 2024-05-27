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

import com.google.common.collect.ImmutableSet;
import lombok.SneakyThrows;
import org.opensearch.core.common.ParsingException;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.index.query.model.HNSWAlgoQueryParameters;
import org.opensearch.knn.index.query.model.AlgoQueryParameters;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import static org.opensearch.core.xcontent.XContentParserUtils.parseFieldsValue;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.index.query.KNNQueryBuilder.EF_SEARCH_FIELD;
import static org.opensearch.knn.index.query.KNNQueryBuilder.NAME;

public final class KNNXParserUtil {

    private static final Set<String> VALID_METHOD_PARAMETERS = ImmutableSet.of(METHOD_PARAMETER_EF_SEARCH);

    public static Map<String, Object> parseJsonObject(XContentParser parser) throws IOException {
        if (parser.currentToken() != XContentParser.Token.START_OBJECT) {
            throw new ParsingException(
                parser.getTokenLocation(),
                "[" + NAME + "] Error parsing json. current token should be START_OBJECT"
            );
        }

        String fieldName = null;
        XContentParser.Token token;
        final Map<String, Object> fieldToValueMap = new HashMap<>();
        while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
            if (token == XContentParser.Token.FIELD_NAME) {
                fieldName = parser.currentName();
            } else {
                assert fieldName != null;
                fieldToValueMap.put(fieldName, parseFieldsValue(parser));
            }
        }
        return fieldToValueMap;
    }

    @SneakyThrows(IOException.class)
    public static AlgoQueryParameters parseMethodParameters(XContentParser parser) {
        final Map<String, Object> parameters = parseJsonObject(parser);
        if (parameters.isEmpty()) {
            throw new IllegalArgumentException("[" + NAME + "] method_parameter cannot be empty");
        }

        for (String jsonkey : parameters.keySet()) {
            if (!VALID_METHOD_PARAMETERS.contains(jsonkey)) {
                throw new IllegalArgumentException("[" + NAME + "] unknown parameter " + jsonkey + " found.");
            }
        }

        return Optional.ofNullable((Integer) parameters.get(METHOD_PARAMETER_EF_SEARCH))
            .filter(ef -> EF_SEARCH_FIELD.match(METHOD_PARAMETER_EF_SEARCH, parser.getDeprecationHandler()))
            .map(ef -> HNSWAlgoQueryParameters.builder().efSearch(ef).build())
            .orElse(null);
    }

    @SneakyThrows(IOException.class)
    public static void unParseMethodParameters(XContentBuilder xContentBuilder, AlgoQueryParameters algoQueryParameters) {
        final HNSWAlgoQueryParameters hnswAlgoParameters = Optional.ofNullable(algoQueryParameters)
            .filter(HNSWAlgoQueryParameters.class::isInstance)
            .map(HNSWAlgoQueryParameters.class::cast)
            .orElse(null);

        if (hnswAlgoParameters != null && hnswAlgoParameters.getEfSearch().isPresent()) {
            xContentBuilder.startObject(METHOD_PARAMETER);
            xContentBuilder.field(EF_SEARCH_FIELD.getPreferredName(), hnswAlgoParameters.getEfSearch().get());
            xContentBuilder.endObject();
        }
    }
}

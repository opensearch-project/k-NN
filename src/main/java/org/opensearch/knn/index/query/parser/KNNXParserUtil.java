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
import org.opensearch.core.common.ParsingException;
import org.opensearch.core.xcontent.XContentParser;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static org.opensearch.core.xcontent.XContentParserUtils.parseFieldsValue;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
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
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.index.query.request.MethodParameter;

import java.util.Collections;
import java.util.Map;

public class LuceneHNSWContext implements EngineSpecificMethodContext {

    private final Map<String, Parameter<?>> supportedMethodParameters = ImmutableMap.<String, Parameter<?>>builder()
        .put(MethodParameter.EF_SEARCH.getName(), new Parameter.IntegerParameter(MethodParameter.EF_SEARCH.getName(), null, value -> true))
        .build();

    @Override
    public Map<String, Parameter<?>> supportedMethodParameters(QueryContext ctx) {
        if (ctx.queryType.isRadialSearch()) {
            // return empty map if radial search is true
            return Collections.emptyMap();
        }
        // Return the supported method parameters for non-radial cases
        return supportedMethodParameters;
    }
}

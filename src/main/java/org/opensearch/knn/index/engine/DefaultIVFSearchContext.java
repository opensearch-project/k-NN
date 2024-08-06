/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.index.engine.model.QueryContext;
import org.opensearch.knn.index.query.request.MethodParameter;

import java.util.Map;

public final class DefaultIVFSearchContext implements KNNLibrarySearchContext {

    private final Map<String, Parameter<?>> supportedMethodParameters = ImmutableMap.<String, Parameter<?>>builder()
        .put(MethodParameter.NPROBE.getName(), new Parameter.IntegerParameter(MethodParameter.NPROBE.getName(), null, value -> true))
        .build();

    @Override
    public Map<String, Parameter<?>> supportedMethodParameters(QueryContext context) {
        return supportedMethodParameters;
    }
}

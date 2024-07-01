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

package org.opensearch.knn.index.util;

import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.index.Parameter;
import org.opensearch.knn.index.query.request.MethodParameter;

import java.util.Map;

public final class DefaultIVFContext implements EngineSpecificMethodContext {

    private final Map<String, Parameter<?>> supportedMethodParameters = ImmutableMap.<String, Parameter<?>>builder()
        .put(MethodParameter.NPROBE.getName(), new Parameter.IntegerParameter(MethodParameter.NPROBE.getName(), null, value -> true))
        .build();

    @Override
    public Map<String, Parameter<?>> supportedMethodParameters(QueryContext context) {
        return supportedMethodParameters;
    }
}

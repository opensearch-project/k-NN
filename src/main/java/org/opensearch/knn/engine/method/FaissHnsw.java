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

package org.opensearch.knn.engine.method;

import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.index.Parameter;
import org.opensearch.knn.index.query.request.MethodParameter;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

public class FaissHnsw implements EngineSpecificMethodContext {

    private final Map<String, Parameter<?>> supportedMethodParameters = ImmutableMap.<String, Parameter<?>>builder()
        .put(MethodParameter.EF_SEARCH.getName(), new Parameter.IntegerParameter(MethodParameter.EF_SEARCH.getName(), null, value -> true))
        .build();

    @Override
    public String engine() {
        return FAISS_NAME;
    }

    @Override
    public String method() {
        return METHOD_HNSW;
    }

    @Override
    public Map<String, Parameter<?>> supportedMethodParameters() {
        return supportedMethodParameters;
    }
}

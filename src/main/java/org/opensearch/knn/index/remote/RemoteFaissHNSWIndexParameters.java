/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import lombok.experimental.SuperBuilder;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.common.KNNConstants;

import java.io.IOException;

import static org.opensearch.knn.index.remote.KNNRemoteConstants.ALGORITHM_PARAMETERS;

@SuperBuilder
public class RemoteFaissHNSWIndexParameters extends RemoteIndexParameters {
    int m;
    int efConstruction;
    int efSearch;

    @Override
    void addAlgorithmParameters(XContentBuilder builder) throws IOException {
        builder.startObject(ALGORITHM_PARAMETERS);
        builder.field(KNNConstants.METHOD_PARAMETER_M, m);
        builder.field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction);
        builder.field(KNNConstants.METHOD_PARAMETER_EF_SEARCH, efSearch);
        builder.endObject();
    }
}

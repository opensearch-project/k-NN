/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.remoteindexbuild.model;

import lombok.experimental.SuperBuilder;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.remoteindexbuild.constants.KNNRemoteConstants;

import java.io.IOException;

@SuperBuilder
public class RemoteFaissHNSWIndexParameters extends RemoteIndexParameters {

    int m;
    int efConstruction;
    int efSearch;

    @Override
    void addAlgorithmParameters(XContentBuilder builder) throws IOException {
        builder.startObject(KNNRemoteConstants.ALGORITHM_PARAMETERS);
        builder.field(KNNRemoteConstants.METHOD_PARAMETER_M, m);
        builder.field(KNNRemoteConstants.METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction);
        builder.field(KNNRemoteConstants.METHOD_PARAMETER_EF_SEARCH, efSearch);
        builder.endObject();
    }
}

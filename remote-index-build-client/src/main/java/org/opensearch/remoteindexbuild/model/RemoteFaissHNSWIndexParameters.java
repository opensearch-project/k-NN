/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.remoteindexbuild.model;

import lombok.Getter;
import lombok.experimental.SuperBuilder;
import org.opensearch.core.xcontent.XContentBuilder;

import java.io.IOException;

import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.ALGORITHM_PARAMETERS;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.METHOD_PARAMETER_M;

@SuperBuilder
@Getter
public class RemoteFaissHNSWIndexParameters extends RemoteIndexParameters {
    int m;
    int efConstruction;
    int efSearch;

    @Override
    void addAlgorithmParameters(XContentBuilder builder) throws IOException {
        builder.startObject(ALGORITHM_PARAMETERS);
        builder.field(METHOD_PARAMETER_M, m);
        builder.field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction);
        builder.field(METHOD_PARAMETER_EF_SEARCH, efSearch);
        builder.endObject();
    }
}

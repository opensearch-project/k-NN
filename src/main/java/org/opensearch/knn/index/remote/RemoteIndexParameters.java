/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import lombok.experimental.SuperBuilder;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;

import java.io.IOException;

import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.ALGORITHM;

@SuperBuilder
public abstract class RemoteIndexParameters implements ToXContentObject {
    String spaceType;
    String algorithm;

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field(METHOD_PARAMETER_SPACE_TYPE, spaceType);
        builder.field(ALGORITHM, algorithm);
        addAlgorithmParameters(builder);
        builder.endObject();
        return builder;
    }

    abstract void addAlgorithmParameters(XContentBuilder builder) throws IOException;

}

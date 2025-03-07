/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.remoteindexbuild.model;

import lombok.experimental.SuperBuilder;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.remoteindexbuild.constants.KNNRemoteConstants;

import java.io.IOException;

@SuperBuilder
public abstract class RemoteIndexParameters implements ToXContentObject {

    String spaceType;
    String algorithm;

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field(KNNRemoteConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType);
        builder.field(KNNRemoteConstants.ALGORITHM, algorithm);
        addAlgorithmParameters(builder);
        builder.endObject();
        return builder;
    }

    abstract void addAlgorithmParameters(XContentBuilder builder) throws IOException;

}

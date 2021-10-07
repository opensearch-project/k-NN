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

package org.opensearch.knn.plugin.transport;

import org.opensearch.action.support.nodes.BaseNodeResponse;
import org.opensearch.common.io.stream.StreamInput;

import java.io.IOException;

public class RemoveFromCacheNodeResponse extends BaseNodeResponse {

    protected RemoveFromCacheNodeResponse(StreamInput in) throws IOException {
        super(in);
    }
}

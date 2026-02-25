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

import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;

public class GetModelRequestTests extends KNNTestCase {
    public void testStreams() throws IOException {
        String modelId = "test-model";
        GetModelRequest getModelRequest = new GetModelRequest(modelId);
        BytesStreamOutput streamOutput = new BytesStreamOutput();
        getModelRequest.writeTo(streamOutput);
        GetModelRequest getModelRequestCopy = new GetModelRequest(streamOutput.bytes().streamInput());
        assertEquals(getModelRequest.getModelID(), getModelRequestCopy.getModelID());
    }
}

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

public class DeleteModelRequestTests extends KNNTestCase {
    public void testStreams() throws IOException {
        String modelId = "test-model";
        DeleteModelRequest deleteModelRequest = new DeleteModelRequest(modelId);
        BytesStreamOutput streamOutput = new BytesStreamOutput();
        deleteModelRequest.writeTo(streamOutput);
        DeleteModelRequest deleteModelRequestCopy = new DeleteModelRequest(streamOutput.bytes().streamInput());
        assertEquals(deleteModelRequest.getModelID(), deleteModelRequestCopy.getModelID());
    }
}

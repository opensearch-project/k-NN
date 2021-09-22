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

package org.opensearch.knn.indices;

import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;

public class ModelStateTests extends KNNTestCase {

    public void testStreams() throws IOException {
        BytesStreamOutput streamOutput = new BytesStreamOutput();
        ModelState original = ModelState.CREATED;

        original.writeTo(streamOutput);

        ModelState modelStateCopy = ModelState.readFrom(streamOutput.bytes().streamInput());

        assertEquals(original, modelStateCopy);
    }

    public void testGetModelState() {
        assertEquals(ModelState.CREATED, ModelState.getModelState(ModelState.CREATED.getName()));
    }
}

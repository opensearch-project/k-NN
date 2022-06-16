/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin;

import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class BlockedModelIdsTests extends OpenSearchTestCase {

    public void testAdd() {
        List<String> modelIds = new ArrayList<>();
        BlockedModelIds testBlockedModelIds = new BlockedModelIds(modelIds);
        String testModelId = "test-model-id";
        testBlockedModelIds.add(testModelId);
        assertTrue(testBlockedModelIds.contains(testModelId));
    }

    public void testRemove() {
        List<String> modelIds = new ArrayList<>();
        String testModelId = "test-model-id";
        modelIds.add(testModelId);
        BlockedModelIds testBlockedModelIds = new BlockedModelIds(modelIds);

        assertTrue(testBlockedModelIds.contains(testModelId));
        testBlockedModelIds.remove(testModelId);
        assertFalse(testBlockedModelIds.contains(testModelId));
    }

    public void testContains() {
        List<String> modelIds = new ArrayList<>();
        String testModelId = "test-model-id";
        modelIds.add(testModelId);

        BlockedModelIds testBlockedModelIds = new BlockedModelIds(modelIds);
        assertTrue(testBlockedModelIds.contains(testModelId));
    }

    public void testStreams() throws IOException {
        List<String> modelIds = new ArrayList<>();
        String testModelId = "test-model-id";
        modelIds.add(testModelId);
        BlockedModelIds testBlockedModelIds = new BlockedModelIds(modelIds);

        BytesStreamOutput streamOutput = new BytesStreamOutput();
        testBlockedModelIds.writeTo(streamOutput);

        BlockedModelIds testBlockedModelIdsCopy = new BlockedModelIds(streamOutput.bytes().streamInput());

        assertEquals(testBlockedModelIds.size(), testBlockedModelIdsCopy.size());
        assertEquals(testBlockedModelIds.getBlockedModelIds().get(0), testBlockedModelIdsCopy.getBlockedModelIds().get(0));
    }

}

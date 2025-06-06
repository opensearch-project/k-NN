/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.junit.Test;
import org.opensearch.knn.index.query.memoryoptsearch.BitSetParentIdGrouper;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

public class BitSetParentIdGrouperTests {
    @Test
    public void testEmptyCase() {
        // When null parent ids were given, then it must return null
        BitSetParentIdGrouper grouper = BitSetParentIdGrouper.createGrouper(null);
        assertNull(grouper);

        // Also, when an empty parent ids were given, null must be returned.
        int[] emptyIds = new int[0];
        grouper = BitSetParentIdGrouper.createGrouper(emptyIds);
        assertNull(grouper);
    }

    @Test
    public void testWhenNotEmptyParentIds() {
        // Meaning
        // - 0-3's parent doc id=4
        // - after the previous parent doc id=4, child docs=5 to 99, their parent doc id=100
        // - 101-103's parent doc id=104
        // - 105-1023's parent doc id=1024
        // - etc
        final int[] parentIds = new int[] { 4, 100, 104, 1024, 11000 };
        Map<Integer, Integer> expectedParentIds = new HashMap<>();
        for (int i = 0, lastParentId = -1; i < parentIds.length; i++) {
            final int parentId = parentIds[i];
            for (int childId = lastParentId + 1; childId < parentId; childId++) {
                expectedParentIds.put(childId, parentId);
            }
            lastParentId = parentId;
        }

        BitSetParentIdGrouper grouper = BitSetParentIdGrouper.createGrouper(parentIds);
        assertNotNull(grouper);

        for (Map.Entry<Integer, Integer> entry : expectedParentIds.entrySet()) {
            final int expectedParentId = entry.getValue();
            final int actualParentId = grouper.getGroupId(entry.getKey());
            assertEquals(expectedParentId, actualParentId);
        }
    }

    @Test
    public void testWhenInvalidChildId() {
        // Meaning
        // - 0-3's parent doc id=4
        // - 5-99's parent doc id=100
        // - 101-103's parent doc id=104
        // - 105-1023's parent doc id=1024
        // - etc
        final int[] parentIds = new int[] { 4, 100, 104, 1024, 11000 };
        BitSetParentIdGrouper grouper = BitSetParentIdGrouper.createGrouper(parentIds);

        // When an invalid child id was given, which is greater than the maximum parent id 11000.
        final int invalidGroupId = grouper.getGroupId(11000 + 1);

        // Match with FAISS's behavior.
        assertEquals(invalidGroupId, Integer.MAX_VALUE);
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.backward_codecs.KNN9120Codec;

import org.opensearch.knn.KNNTestCase;

public class ParentChildHelperTests extends KNNTestCase {

    public void testGetParentField() {
        assertEquals("parent.to", ParentChildHelper.getParentField("parent.to.child"));
        assertEquals("parent", ParentChildHelper.getParentField("parent.to"));
        assertNull(ParentChildHelper.getParentField("child"));
        assertNull(ParentChildHelper.getParentField(""));
        assertNull(ParentChildHelper.getParentField(null));
    }

    public void testGetChildField() {
        assertEquals("child", ParentChildHelper.getChildField("parent.to.child"));
        assertNull(ParentChildHelper.getChildField(null));
        assertNull(ParentChildHelper.getChildField("child"));
    }

    public void testConstructSiblingField() {
        assertEquals("parent.to.sibling", ParentChildHelper.constructSiblingField("parent.to.child", "sibling"));
        assertEquals("sibling", ParentChildHelper.constructSiblingField("parent", "sibling"));
    }

    public void testSplitPath() {
        String[] path = ParentChildHelper.splitPath("parent.to.child");
        assertEquals(3, path.length);
        assertEquals("parent", path[0]);
        assertEquals("to", path[1]);
        assertEquals("child", path[2]);

        path = ParentChildHelper.splitPath("parent");
        assertEquals(1, path.length);
        assertEquals("parent", path[0]);
    }
}

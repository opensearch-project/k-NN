/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.junit.Test;
import org.opensearch.knn.KNNTestCase;

import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

public class FaissEmptyIndexTests extends KNNTestCase {

    @Test
    public void testInstanceIsSingleton() {
        assertSame(FaissEmptyIndex.INSTANCE, FaissEmptyIndex.INSTANCE);
    }

    @Test
    public void testGetVectorEncodingThrows() {
        UnsupportedOperationException ex = assertThrows(
            UnsupportedOperationException.class,
            () -> FaissEmptyIndex.INSTANCE.getVectorEncoding()
        );
        assertTrue(ex.getMessage().contains(FaissEmptyIndex.class.getSimpleName()));
        assertTrue(ex.getMessage().contains("does not support this operation"));
    }

    @Test
    public void testGetFloatValuesThrows() {
        UnsupportedOperationException ex = assertThrows(
            UnsupportedOperationException.class,
            () -> FaissEmptyIndex.INSTANCE.getFloatValues(null)
        );
        assertTrue(ex.getMessage().contains(FaissEmptyIndex.class.getSimpleName()));
        assertTrue(ex.getMessage().contains("does not support this operation"));
    }

    @Test
    public void testGetByteValuesThrows() {
        UnsupportedOperationException ex = assertThrows(
            UnsupportedOperationException.class,
            () -> FaissEmptyIndex.INSTANCE.getByteValues(null)
        );
        assertTrue(ex.getMessage().contains(FaissEmptyIndex.class.getSimpleName()));
        assertTrue(ex.getMessage().contains("does not support this operation"));
    }

    @Test
    public void testIndexTypeIsNull() {
        assertEquals(FaissIndex.NULL_INDEX_TYPE, FaissEmptyIndex.INSTANCE.getIndexType());
    }
}

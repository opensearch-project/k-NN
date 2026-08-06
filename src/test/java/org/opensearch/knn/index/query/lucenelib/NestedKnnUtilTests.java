/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import junit.framework.TestCase;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.util.BitSet;

import java.io.IOException;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class NestedKnnUtilTests extends TestCase {

    public void testHasNoParentDocs_whenBitSetNull_thenReturnTrue() throws IOException {
        BitSetProducer parentFilter = mock(BitSetProducer.class);
        LeafReaderContext context = mock(LeafReaderContext.class);

        when(parentFilter.getBitSet(context)).thenReturn(null);

        assertTrue(NestedKnnUtil.hasNoParentDocs(parentFilter, context));
    }

    public void testHasNoParentDocs_whenBitSetExists_thenReturnFalse() throws IOException {
        BitSetProducer parentFilter = mock(BitSetProducer.class);
        LeafReaderContext context = mock(LeafReaderContext.class);
        BitSet bitSet = mock(BitSet.class);

        when(parentFilter.getBitSet(context)).thenReturn(bitSet);

        assertFalse(NestedKnnUtil.hasNoParentDocs(parentFilter, context));
    }
}

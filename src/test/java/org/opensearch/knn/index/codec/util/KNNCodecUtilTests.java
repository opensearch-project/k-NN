/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import junit.framework.TestCase;
import lombok.SneakyThrows;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.index.codec.transfer.VectorTransfer;

import java.util.Arrays;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class KNNCodecUtilTests extends TestCase {
    @SneakyThrows
    public void testGetPair_whenCalled_thenReturn() {
        long liveDocCount = 1l;
        int[] docId = { 2 };
        long vectorAddress = 3l;
        int dimension = 4;
        BytesRef bytesRef = new BytesRef();

        BinaryDocValues binaryDocValues = mock(BinaryDocValues.class);
        when(binaryDocValues.cost()).thenReturn(liveDocCount);
        when(binaryDocValues.nextDoc()).thenReturn(docId[0], NO_MORE_DOCS);
        when(binaryDocValues.binaryValue()).thenReturn(bytesRef);

        VectorTransfer vectorTransfer = mock(VectorTransfer.class);
        when(vectorTransfer.getVectorAddress()).thenReturn(vectorAddress);
        when(vectorTransfer.getDimension()).thenReturn(dimension);

        // Run
        KNNCodecUtil.VectorBatch batch = KNNCodecUtil.getVectorBatch(binaryDocValues, vectorTransfer, false);

        // Verify
        verify(vectorTransfer).init(liveDocCount);
        verify(vectorTransfer).transfer(any(BytesRef.class));
        verify(vectorTransfer).close();

        assertTrue(Arrays.equals(docId, batch.docs));
        assertEquals(vectorAddress, batch.getVectorAddress());
        assertEquals(dimension, batch.getDimension());
    }
}

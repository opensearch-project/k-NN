/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.SneakyThrows;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.KNNTestCase;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;

public class FaissIndexTests extends KNNTestCase {

    @SneakyThrows
    public void testLoad_whenNullSection_thenReturnsFaissEmptyIndex() {
        IndexInput mockInput = mock(IndexInput.class);
        doAnswer(invocation -> {
            byte[] buf = invocation.getArgument(0);
            byte[] nullBytes = "null".getBytes();
            System.arraycopy(nullBytes, 0, buf, 0, 4);
            return null;
        }).when(mockInput).readBytes(any(byte[].class), any(int.class), any(int.class));

        FaissIndex result = FaissIndex.load(mockInput);
        assertSame(FaissEmptyIndex.INSTANCE, result);
    }

}

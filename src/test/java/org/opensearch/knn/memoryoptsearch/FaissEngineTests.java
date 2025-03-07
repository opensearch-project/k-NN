/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class FaissEngineTests extends KNNTestCase {
    public void testFaissEngineToReturnSearcher() throws IOException {
        final VectorSearcherFactory factory = KNNEngine.FAISS.getVectorSearcherFactory();
        assertNotNull(factory);

        final IndexInput mockIndexInput = mock(IndexInput.class);
        final Directory mockDirectory = mock(Directory.class);
        when(mockDirectory.openInput(any(), any())).thenReturn(mockIndexInput);
        final String fileName = "_0_165_target_field.faiss";
        try (VectorSearcher searcher = factory.createVectorSearcher(mockDirectory, fileName)) {
            assertNotNull(searcher);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void testNonFaissEngineToReturnNullSearcher() {
        assertNull(KNNEngine.LUCENE.getVectorSearcherFactory());
        assertNull(KNNEngine.NMSLIB.getVectorSearcherFactory());
    }
}

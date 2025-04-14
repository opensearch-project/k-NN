/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.mockito.Mock;
import org.opensearch.knn.KNNTestCase;

import static org.mockito.Mockito.verify;

public class DerivedSourceReadersTests extends KNNTestCase {

    @Mock
    private KnnVectorsReader mockKnnVectorsReader;
    @Mock
    private DocValuesProducer mockDocValuesProducer;

    private DerivedSourceReaders readers;

    @SneakyThrows
    public void testInitialReferenceCount() {
        readers = new DerivedSourceReaders(mockKnnVectorsReader, mockDocValuesProducer);

        // Initial reference count is 1, so closing once should trigger actual close
        readers.close();

        verify(mockKnnVectorsReader).close();
        verify(mockDocValuesProducer).close();
    }

    @SneakyThrows
    public void testNullReaders() {
        // Test with null readers to ensure no NPE
        DerivedSourceReaders nullReaders = new DerivedSourceReaders(null, null);
        nullReaders.close(); // Should not throw any exception
    }
}

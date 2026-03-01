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

import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

public class DerivedSourceReadersTests extends KNNTestCase {

    @Mock
    private KnnVectorsReader mockKnnVectorsReader;
    @Mock
    private DocValuesProducer mockDocValuesProducer;

    private DerivedSourceReaders readers;

    @SneakyThrows
    public void testClose() {
        readers = new DerivedSourceReaders(mockKnnVectorsReader, mockDocValuesProducer);
        readers.close();
        verify(mockKnnVectorsReader).close();
        verify(mockDocValuesProducer).close();
    }

    @SneakyThrows
    public void testCloneDoesNotClose() {
        readers = new DerivedSourceReaders(mockKnnVectorsReader, mockDocValuesProducer);
        DerivedSourceReaders clone = readers.clone();
        clone.close(); // should be no-op
        verify(mockKnnVectorsReader, never()).close();
        verify(mockDocValuesProducer, never()).close();
    }

    @SneakyThrows
    public void testGetMergeInstanceDoesNotClose() {
        readers = new DerivedSourceReaders(mockKnnVectorsReader, mockDocValuesProducer);
        DerivedSourceReaders mergeInstance = readers.getMergeInstance();
        mergeInstance.close(); // no-op
        verify(mockKnnVectorsReader, never()).close();
        verify(mockDocValuesProducer, never()).close();
    }

    @SneakyThrows
    public void testNullReaders() {
        expectThrows(AssertionError.class, () -> new DerivedSourceReaders(null, null));
    }
}

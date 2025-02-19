/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.index.StoredFieldVisitor;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class DerivedSourceStoredFieldVisitorTests extends KNNTestCase {

    public void testBinaryField() throws Exception {
        StoredFieldVisitor delegate = mock(StoredFieldVisitor.class);
        doAnswer(invocationOnMock -> null).when(delegate).binaryField(any(), any());
        DerivedSourceVectorInjector derivedSourceVectorInjector = mock(DerivedSourceVectorInjector.class);
        when(derivedSourceVectorInjector.injectVectors(anyInt(), any())).thenReturn(new byte[0]);
        DerivedSourceStoredFieldVisitor derivedSourceStoredFieldVisitor = new DerivedSourceStoredFieldVisitor(
            delegate,
            0,
            derivedSourceVectorInjector
        );

        // When field is not _source, then do not call the injector
        derivedSourceStoredFieldVisitor.binaryField(KNNCodecTestUtil.FieldInfoBuilder.builder("test").build(), null);
        verify(derivedSourceVectorInjector, times(0)).injectVectors(anyInt(), any());
        verify(delegate, times(1)).binaryField(any(), any());

        // When field is not _source, then do call the injector
        derivedSourceStoredFieldVisitor.binaryField(KNNCodecTestUtil.FieldInfoBuilder.builder("_source").build(), null);
        verify(derivedSourceVectorInjector, times(1)).injectVectors(anyInt(), any());
        verify(delegate, times(2)).binaryField(any(), any());
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.StoredFieldVisitor;
import org.opensearch.index.mapper.SourceFieldMapper;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

public class DerivedSourceStoredFieldVisitorTests extends OpenSearchTestCase {

    private static final String TEST_ORIGINAL_VALUE = "original";
    private static final String TEST_TRANSFORMED_VALUE = "transformed";
    private static final String TEST_VALUE = "test";
    private static final String TEST_STRING_VALUE = "test-value";
    private static final int TEST_DOC_ID = 123;
    private static final long TEST_LONG_VALUE = 42L;
    private static final int TEST_INT_VALUE = 42;

    public void testBinaryField_whenSourceField_thenInjectsVectors() throws IOException {
        StoredFieldVisitor delegate = mock(StoredFieldVisitor.class);
        DerivedSourceVectorTransformer transformer = mock(DerivedSourceVectorTransformer.class);
        FieldInfo fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder(SourceFieldMapper.NAME).build();

        byte[] originalValue = TEST_ORIGINAL_VALUE.getBytes();
        byte[] transformedValue = TEST_TRANSFORMED_VALUE.getBytes();
        int documentId = TEST_DOC_ID;

        when(transformer.injectVectors(documentId, originalValue)).thenReturn(transformedValue);

        DerivedSourceStoredFieldVisitor visitor = new DerivedSourceStoredFieldVisitor(delegate, documentId, transformer);

        visitor.binaryField(fieldInfo, originalValue);

        verify(transformer).injectVectors(documentId, originalValue);
        verify(delegate).binaryField(fieldInfo, transformedValue);
    }

    public void testBinaryField_whenNonSourceField_thenDelegatesDirectly() throws IOException {
        StoredFieldVisitor delegate = mock(StoredFieldVisitor.class);
        DerivedSourceVectorTransformer transformer = mock(DerivedSourceVectorTransformer.class);
        FieldInfo fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("other-field").build();

        byte[] value = TEST_VALUE.getBytes();
        int documentId = TEST_DOC_ID;

        DerivedSourceStoredFieldVisitor visitor = new DerivedSourceStoredFieldVisitor(delegate, documentId, transformer);

        visitor.binaryField(fieldInfo, value);

        verify(delegate).binaryField(fieldInfo, value);
        verifyNoInteractions(transformer);
    }

    public void testBinaryField_whenNullValue_thenHandlesGracefully() throws IOException {
        StoredFieldVisitor delegate = mock(StoredFieldVisitor.class);
        DerivedSourceVectorTransformer transformer = mock(DerivedSourceVectorTransformer.class);
        FieldInfo fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("other-field").build();

        DerivedSourceStoredFieldVisitor visitor = new DerivedSourceStoredFieldVisitor(delegate, TEST_DOC_ID, transformer);

        visitor.binaryField(fieldInfo, (byte[]) null);

        verify(delegate).binaryField(fieldInfo, (byte[]) null);
        verifyNoInteractions(transformer);
    }

    public void testNeedsField_delegatesToDelegate() throws IOException {
        StoredFieldVisitor delegate = mock(StoredFieldVisitor.class);
        DerivedSourceVectorTransformer transformer = mock(DerivedSourceVectorTransformer.class);
        FieldInfo fieldInfo = mock(FieldInfo.class);

        when(delegate.needsField(fieldInfo)).thenReturn(StoredFieldVisitor.Status.YES);

        DerivedSourceStoredFieldVisitor visitor = new DerivedSourceStoredFieldVisitor(delegate, TEST_DOC_ID, transformer);

        StoredFieldVisitor.Status result = visitor.needsField(fieldInfo);

        assertEquals(StoredFieldVisitor.Status.YES, result);
        verify(delegate).needsField(fieldInfo);
    }

    public void testStringField_delegatesToDelegate() throws IOException {
        StoredFieldVisitor delegate = mock(StoredFieldVisitor.class);
        DerivedSourceVectorTransformer transformer = mock(DerivedSourceVectorTransformer.class);
        FieldInfo fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("test-field").build();

        DerivedSourceStoredFieldVisitor visitor = new DerivedSourceStoredFieldVisitor(delegate, TEST_DOC_ID, transformer);

        visitor.stringField(fieldInfo, TEST_STRING_VALUE);

        verify(delegate).stringField(fieldInfo, TEST_STRING_VALUE);
    }

    public void testLongField_delegatesToDelegate() throws IOException {
        StoredFieldVisitor delegate = mock(StoredFieldVisitor.class);
        DerivedSourceVectorTransformer transformer = mock(DerivedSourceVectorTransformer.class);
        FieldInfo fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("test-field").build();

        DerivedSourceStoredFieldVisitor visitor = new DerivedSourceStoredFieldVisitor(delegate, TEST_DOC_ID, transformer);

        visitor.longField(fieldInfo, TEST_LONG_VALUE);

        verify(delegate).longField(fieldInfo, TEST_LONG_VALUE);
    }

    public void testIntField_delegatesToDelegate() throws IOException {
        StoredFieldVisitor delegate = mock(StoredFieldVisitor.class);
        DerivedSourceVectorTransformer transformer = mock(DerivedSourceVectorTransformer.class);
        FieldInfo fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("test-field").build();

        DerivedSourceStoredFieldVisitor visitor = new DerivedSourceStoredFieldVisitor(delegate, TEST_DOC_ID, transformer);

        visitor.intField(fieldInfo, TEST_INT_VALUE);

        verify(delegate).intField(fieldInfo, TEST_INT_VALUE);
    }
}

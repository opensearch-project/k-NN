/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SegmentReadState;
import org.junit.Before;
import org.mockito.Mock;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.codec.derivedsource.DerivedSourceLuceneHelper.NO_CHILDREN_INDICATOR;

public class DerivedSourceLuceneHelperTests extends KNNTestCase {
    @Mock
    private FieldInfos fieldInfos;

    @Mock
    private FieldInfo fieldInfo;

    @Mock
    private DerivedSourceReaders derivedSourceReaders;

    @Mock
    private DocValuesProducer docValuesProducer;

    @Mock
    private NumericDocValues numericDocValues;

    private SegmentReadState segmentReadState;
    private DerivedSourceLuceneHelper helper;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        segmentReadState = new SegmentReadState(null, null, fieldInfos, null, null);

        when(fieldInfos.fieldInfo("_primary_term")).thenReturn(fieldInfo);
        when(derivedSourceReaders.getDocValuesProducer()).thenReturn(docValuesProducer);
        when(docValuesProducer.getNumeric(fieldInfo)).thenReturn(numericDocValues);
        helper = new DerivedSourceLuceneHelper(derivedSourceReaders, segmentReadState);
    }

    @SneakyThrows
    public void testGetFirstChild_WhenNoDocumentsBeforeParent() throws IOException {
        int parentDocId = 5;
        int startingPoint = 0;
        when(numericDocValues.advance(startingPoint)).thenReturn(10); // First doc is after parent
        when(numericDocValues.docID()).thenReturn(10, NO_MORE_DOCS);

        int result = helper.getFirstChild(parentDocId, startingPoint);

        assertEquals(0, result);
    }

    @SneakyThrows
    public void testGetFirstChild_WhenNoChildren() {
        int parentDocId = 5;
        int startingPoint = 0;
        when(numericDocValues.advance(startingPoint)).thenReturn(4);
        when(numericDocValues.docID()).thenReturn(4, 4, 4, 5, 5);
        when(numericDocValues.nextDoc()).thenReturn(5);

        int result = helper.getFirstChild(parentDocId, startingPoint);

        assertEquals(NO_CHILDREN_INDICATOR, result);
    }

    @SneakyThrows
    public void testGetFirstChild_WhenChildrenExist() {
        int parentDocId = 10;
        int startingPoint = 0;
        when(numericDocValues.advance(startingPoint)).thenReturn(5);
        when(numericDocValues.docID()).thenReturn(5, 5, 5, 10, 10);
        when(numericDocValues.nextDoc()).thenReturn(10);

        int result = helper.getFirstChild(parentDocId, startingPoint);

        assertEquals(6, result); // Should return previousParentDocId + 1
    }
}

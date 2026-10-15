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

    @SneakyThrows
    public void testGetFirstChild_WhenNoParentAfterNonZeroStartingPoint_thenNoMoreDocs() {
        // Parent 500 has 399 children (101..499); previous parent 100 is before the starting point
        when(docValuesProducer.getNumeric(fieldInfo)).thenAnswer(invocation -> rootDocs(100, 500));

        assertEquals(NO_MORE_DOCS, helper.getFirstChild(500, 350));
    }

    @SneakyThrows
    public void testGetFirstChild_WhenParentHasMoreChildrenThanFirstOffset() {
        // Parent 500 has 399 children (101..499), so the first offset window (350..499) contains no parent
        when(docValuesProducer.getNumeric(fieldInfo)).thenAnswer(invocation -> rootDocs(100, 500));

        assertEquals(101, helper.getFirstChild(500));
    }

    @SneakyThrows
    public void testGetFirstChild_WhenFirstParentHasMoreChildrenThanFirstOffset() {
        // Parent 500 is the first parent in the segment and has 500 children (0..499)
        when(docValuesProducer.getNumeric(fieldInfo)).thenAnswer(invocation -> rootDocs(500));

        assertEquals(0, helper.getFirstChild(500));
    }

    private static NumericDocValues rootDocs(int... docIds) {
        return new NumericDocValues() {
            private int index = -1;

            @Override
            public long longValue() {
                return 1;
            }

            @Override
            public boolean advanceExact(int target) {
                throw new UnsupportedOperationException();
            }

            @Override
            public int docID() {
                if (index < 0) {
                    return -1;
                }
                return index < docIds.length ? docIds[index] : NO_MORE_DOCS;
            }

            @Override
            public int nextDoc() {
                index++;
                return docID();
            }

            @Override
            public int advance(int target) {
                do {
                    index++;
                } while (index < docIds.length && docIds[index] < target);
                return docID();
            }

            @Override
            public long cost() {
                return docIds.length;
            }
        };
    }
}

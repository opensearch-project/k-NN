/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.knn.KNNTestCase;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.junit.Assert;
import org.junit.Before;

import java.io.IOException;

public class KNNVectorScriptDocValuesTests extends KNNTestCase {

    private static final String MOCK_INDEX_FIELD_NAME = "test-index-field-name";
    private static final float[] SAMPLE_VECTOR_DATA = new float[] { 1.0f, 2.0f };
    private static final byte[] SAMPLE_BYTE_VECTOR_DATA = new byte[] { 1, 2 };
    private KNNVectorScriptDocValues<?> scriptDocValues;
    private Directory directory;
    private DirectoryReader reader;
    private Class<? extends DocIdSetIterator> valuesClass;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        directory = newDirectory();
        valuesClass = randomFrom(BinaryDocValues.class, ByteVectorValues.class, FloatVectorValues.class);
        createKNNVectorDocument(directory, valuesClass);
        reader = DirectoryReader.open(directory);
        LeafReader leafReader = reader.getContext().leaves().get(0).reader();
        DocIdSetIterator vectorValues;
        if (BinaryDocValues.class.equals(valuesClass)) {
            vectorValues = DocValues.getBinary(leafReader, MOCK_INDEX_FIELD_NAME);
        } else if (ByteVectorValues.class.equals(valuesClass)) {
            vectorValues = leafReader.getByteVectorValues(MOCK_INDEX_FIELD_NAME);
        } else {
            vectorValues = leafReader.getFloatVectorValues(MOCK_INDEX_FIELD_NAME);
        }

        scriptDocValues = KNNVectorScriptDocValues.create(vectorValues, MOCK_INDEX_FIELD_NAME, VectorDataType.FLOAT);
    }

    private void createKNNVectorDocument(Directory directory, Class<? extends DocIdSetIterator> valuesClass) throws IOException {
        IndexWriterConfig conf = newIndexWriterConfig(new MockAnalyzer(random()));
        IndexWriter writer = new IndexWriter(directory, conf);
        Document knnDocument = new Document();
        Field field;
        if (BinaryDocValues.class.equals(valuesClass)) {
            field = new BinaryDocValuesField(
                MOCK_INDEX_FIELD_NAME,
                new VectorField(MOCK_INDEX_FIELD_NAME, SAMPLE_VECTOR_DATA, new FieldType()).binaryValue()
            );
        } else if (ByteVectorValues.class.equals(valuesClass)) {
            field = new KnnByteVectorField(MOCK_INDEX_FIELD_NAME, SAMPLE_BYTE_VECTOR_DATA);
        } else {
            field = new KnnFloatVectorField(MOCK_INDEX_FIELD_NAME, SAMPLE_VECTOR_DATA);
        }

        knnDocument.add(field);
        writer.addDocument(knnDocument);
        writer.commit();
        writer.close();
    }

    @Override
    public void tearDown() throws Exception {
        super.tearDown();
        reader.close();
        directory.close();
    }

    @SuppressWarnings("unchecked")
    public void testGetValue() throws IOException {
        scriptDocValues.setNextDocId(0);
        if (ByteVectorValues.class.equals(valuesClass)) {
            Assert.assertArrayEquals(SAMPLE_BYTE_VECTOR_DATA, ((KNNVectorScriptDocValues<byte[]>) scriptDocValues).getValue());
        } else {
            Assert.assertArrayEquals(SAMPLE_VECTOR_DATA, ((KNNVectorScriptDocValues<float[]>) scriptDocValues).getValue(), 0.1f);
        }
    }

    // Test getValue without calling setNextDocId
    public void testGetValueFails() throws IOException {
        expectThrows(IllegalStateException.class, () -> scriptDocValues.getValue());
    }

    public void testSize() throws IOException {
        Assert.assertEquals(0, scriptDocValues.size());
        scriptDocValues.setNextDocId(0);
        Assert.assertEquals(1, scriptDocValues.size());
    }

    public void testGet() throws IOException {
        expectThrows(UnsupportedOperationException.class, () -> scriptDocValues.get(0));
    }

    public void testUnsupportedValues() throws IOException {
        expectThrows(
            IllegalArgumentException.class,
            () -> KNNVectorScriptDocValues.create(DocValues.emptyNumeric(), MOCK_INDEX_FIELD_NAME, VectorDataType.FLOAT)
        );
    }

    public void testEmptyValues() throws IOException {
        KNNVectorScriptDocValues values = KNNVectorScriptDocValues.emptyValues(MOCK_INDEX_FIELD_NAME, VectorDataType.FLOAT);
        assertEquals(0, values.size());
        scriptDocValues.setNextDocId(0);
        assertEquals(0, values.size());
    }
}

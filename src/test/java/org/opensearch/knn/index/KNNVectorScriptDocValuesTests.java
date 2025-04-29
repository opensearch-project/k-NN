/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.*;
import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.KNNTestCase;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.store.Directory;
import org.junit.Assert;
import org.junit.Before;
import org.junit.After;
import org.junit.Test;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfFloatsSerializer;

import java.io.IOException;

public class KNNVectorScriptDocValuesTests extends KNNTestCase {

    private static final String MOCK_INDEX_FIELD_NAME = "test-index-field-name";
    private static final float[] SAMPLE_VECTOR_DATA = new float[] { 1.0f, 2.0f };
    private static final byte[] SAMPLE_BYTE_VECTOR_DATA = new byte[] { 1, 2 };

    private Directory directory;
    private DirectoryReader reader;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        directory = newDirectory();
    }

    @After
    public void tearDown() throws Exception {
        super.tearDown();
        if (reader != null) {
            reader.close();
        }
        if (directory != null) {
            directory.close();
        }
    }

    /** Test for Float Vector Values */
    @Test
    public void testFloatVectorValues() throws IOException {
        createKNNVectorDocument(directory, FloatVectorValues.class);
        reader = DirectoryReader.open(directory);
        LeafReader leafReader = reader.leaves().get(0).reader();

        // Separate scriptDocValues instance for this test
        KNNVectorScriptDocValues scriptDocValues = KNNVectorScriptDocValues.create(
            leafReader.getFloatVectorValues(MOCK_INDEX_FIELD_NAME),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        );

        scriptDocValues.setNextDocId(0);
        Assert.assertArrayEquals(SAMPLE_VECTOR_DATA, scriptDocValues.getValue(), 0.1f);
    }

    /** Test for Byte Vector Values */
    @Test
    public void testByteVectorValues() throws IOException {
        createKNNVectorDocument(directory, ByteVectorValues.class);
        reader = DirectoryReader.open(directory);
        LeafReader leafReader = reader.leaves().get(0).reader();

        KNNVectorScriptDocValues scriptDocValues = KNNVectorScriptDocValues.create(
            leafReader.getByteVectorValues(MOCK_INDEX_FIELD_NAME),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.BYTE
        );

        scriptDocValues.setNextDocId(0);
        Assert.assertArrayEquals(new float[] { SAMPLE_BYTE_VECTOR_DATA[0], SAMPLE_BYTE_VECTOR_DATA[1] }, scriptDocValues.getValue(), 0.1f);
    }

    /** Test for Binary Vector Values */
    @Test
    public void testBinaryVectorValues() throws IOException {
        createKNNVectorDocument(directory, BinaryDocValues.class);
        reader = DirectoryReader.open(directory);
        LeafReader leafReader = reader.leaves().get(0).reader();

        KNNVectorScriptDocValues scriptDocValues = KNNVectorScriptDocValues.create(
            leafReader.getBinaryDocValues(MOCK_INDEX_FIELD_NAME),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.BINARY
        );

        scriptDocValues.setNextDocId(0);
        Assert.assertNotNull(scriptDocValues.getValue());  // Just checking it's non-null
    }

    /** Ensure getValue() fails without setNextDocId */
    @Test
    public void testGetValueFails() throws IOException {
        createKNNVectorDocument(directory, FloatVectorValues.class);
        reader = DirectoryReader.open(directory);
        LeafReader leafReader = reader.leaves().get(0).reader();

        KNNVectorScriptDocValues scriptDocValues = KNNVectorScriptDocValues.create(
            leafReader.getFloatVectorValues(MOCK_INDEX_FIELD_NAME),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        );

        expectThrows(IllegalStateException.class, () -> scriptDocValues.getValue());
    }

    /** Ensure size() returns expected values */
    @Test
    public void testSize() throws IOException {
        createKNNVectorDocument(directory, FloatVectorValues.class);
        reader = DirectoryReader.open(directory);
        LeafReader leafReader = reader.leaves().get(0).reader();

        KNNVectorScriptDocValues scriptDocValues = KNNVectorScriptDocValues.create(
            leafReader.getFloatVectorValues(MOCK_INDEX_FIELD_NAME),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        );

        Assert.assertEquals(0, scriptDocValues.size());
        scriptDocValues.setNextDocId(0);
        Assert.assertEquals(1, scriptDocValues.size());
    }

    /** Ensure get() throws UnsupportedOperationException */
    @Test
    public void testGet() {
        expectThrows(UnsupportedOperationException.class, () -> {
            KNNVectorScriptDocValues scriptDocValues = KNNVectorScriptDocValues.emptyValues(MOCK_INDEX_FIELD_NAME, VectorDataType.FLOAT);
            scriptDocValues.get(0);
        });
    }

    /** Test unsupported values type */
    @Test
    public void testUnsupportedValues() throws IOException {
        expectThrows(
            IllegalArgumentException.class,
            () -> KNNVectorScriptDocValues.create(DocValues.emptyNumeric(), MOCK_INDEX_FIELD_NAME, VectorDataType.FLOAT)
        );
    }

    /** Ensure empty values case */
    @Test
    public void testEmptyValues() throws IOException {
        KNNVectorScriptDocValues values = KNNVectorScriptDocValues.emptyValues(MOCK_INDEX_FIELD_NAME, VectorDataType.FLOAT);
        assertEquals(0, values.size());
    }

    private void createKNNVectorDocument(Directory directory, Class<?> valuesClass) throws IOException {
        IndexWriterConfig conf = newIndexWriterConfig(new MockAnalyzer(random()));
        IndexWriter writer = new IndexWriter(directory, conf);
        Document knnDocument = new Document();
        Field field;

        if (BinaryDocValues.class.equals(valuesClass)) {
            byte[] vectorBinary = KNNVectorAsCollectionOfFloatsSerializer.INSTANCE.floatToByteArray(SAMPLE_VECTOR_DATA);
            field = new BinaryDocValuesField(MOCK_INDEX_FIELD_NAME, new BytesRef(vectorBinary));
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
}

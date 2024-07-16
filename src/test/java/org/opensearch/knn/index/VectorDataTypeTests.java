/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.SneakyThrows;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.apache.lucene.util.BytesRef;
import org.junit.Assert;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;

public class VectorDataTypeTests extends KNNTestCase {

    private static final String MOCK_FLOAT_INDEX_FIELD_NAME = "test-float-index-field-name";
    private static final String MOCK_BYTE_INDEX_FIELD_NAME = "test-byte-index-field-name";
    private static final float[] SAMPLE_FLOAT_VECTOR_DATA = new float[] { 10.0f, 25.0f };
    private static final byte[] SAMPLE_BYTE_VECTOR_DATA = new byte[] { 10, 25 };
    private Directory directory;
    private DirectoryReader reader;

    @SneakyThrows
    public void testGetDocValuesWithFloatVectorDataType() {
        KNNVectorScriptDocValues scriptDocValues = getKNNFloatVectorScriptDocValues();

        scriptDocValues.setNextDocId(0);
        Assert.assertArrayEquals(SAMPLE_FLOAT_VECTOR_DATA, scriptDocValues.getValue(), 0.1f);

        reader.close();
        directory.close();
    }

    @SneakyThrows
    public void testGetDocValuesWithByteVectorDataType() {
        KNNVectorScriptDocValues scriptDocValues = getKNNByteVectorScriptDocValues();

        scriptDocValues.setNextDocId(0);
        Assert.assertArrayEquals(SAMPLE_FLOAT_VECTOR_DATA, scriptDocValues.getValue(), 0.1f);

        reader.close();
        directory.close();
    }

    @SneakyThrows
    private KNNVectorScriptDocValues getKNNFloatVectorScriptDocValues() {
        directory = newDirectory();
        createKNNFloatVectorDocument(directory);
        reader = DirectoryReader.open(directory);
        LeafReaderContext leafReaderContext = reader.getContext().leaves().get(0);
        return KNNVectorScriptDocValues.create(
            leafReaderContext.reader().getBinaryDocValues(VectorDataTypeTests.MOCK_FLOAT_INDEX_FIELD_NAME),
            VectorDataTypeTests.MOCK_FLOAT_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        );
    }

    @SneakyThrows
    private KNNVectorScriptDocValues getKNNByteVectorScriptDocValues() {
        directory = newDirectory();
        createKNNByteVectorDocument(directory);
        reader = DirectoryReader.open(directory);
        LeafReaderContext leafReaderContext = reader.getContext().leaves().get(0);
        return KNNVectorScriptDocValues.create(
            leafReaderContext.reader().getBinaryDocValues(VectorDataTypeTests.MOCK_BYTE_INDEX_FIELD_NAME),
            VectorDataTypeTests.MOCK_BYTE_INDEX_FIELD_NAME,
            VectorDataType.BYTE
        );
    }

    private void createKNNFloatVectorDocument(Directory directory) throws IOException {
        IndexWriterConfig conf = newIndexWriterConfig(new MockAnalyzer(random()));
        IndexWriter writer = new IndexWriter(directory, conf);
        Document knnDocument = new Document();
        knnDocument.add(
            new BinaryDocValuesField(
                MOCK_FLOAT_INDEX_FIELD_NAME,
                new VectorField(MOCK_FLOAT_INDEX_FIELD_NAME, SAMPLE_FLOAT_VECTOR_DATA, new FieldType()).binaryValue()
            )
        );
        writer.addDocument(knnDocument);
        writer.commit();
        writer.close();
    }

    private void createKNNByteVectorDocument(Directory directory) throws IOException {
        IndexWriterConfig conf = newIndexWriterConfig(new MockAnalyzer(random()));
        IndexWriter writer = new IndexWriter(directory, conf);
        Document knnDocument = new Document();
        knnDocument.add(
            new BinaryDocValuesField(
                MOCK_BYTE_INDEX_FIELD_NAME,
                new VectorField(MOCK_BYTE_INDEX_FIELD_NAME, SAMPLE_BYTE_VECTOR_DATA, new FieldType()).binaryValue()
            )
        );
        writer.addDocument(knnDocument);
        writer.commit();
        writer.close();
    }

    public void testCreateKnnVectorFieldType_whenBinary_thenException() {
        Exception ex = expectThrows(
            IllegalStateException.class,
            () -> VectorDataType.BINARY.createKnnVectorFieldType(1, VectorSimilarityFunction.EUCLIDEAN)
        );
        assertTrue(ex.getMessage().contains("Unsupported method"));
    }

    public void testGetVectorFromBytesRef_whenBinary_thenException() {
        byte[] vector = { 1, 2, 3 };
        float[] expected = { 1, 2, 3 };
        BytesRef bytesRef = new BytesRef(vector);
        assertArrayEquals(expected, VectorDataType.BINARY.getVectorFromBytesRef(bytesRef), 0.01f);
    }
}

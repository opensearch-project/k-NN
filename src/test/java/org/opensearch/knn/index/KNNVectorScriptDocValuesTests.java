/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.index.LeafReaderContext;
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
    private KNNVectorScriptDocValues scriptDocValues;
    private Directory directory;
    private DirectoryReader reader;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        directory = newDirectory();
        createKNNVectorDocument(directory);
        reader = DirectoryReader.open(directory);
        LeafReaderContext leafReaderContext = reader.getContext().leaves().get(0);
        scriptDocValues = new KNNVectorScriptDocValues(
            leafReaderContext.reader().getBinaryDocValues(MOCK_INDEX_FIELD_NAME),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        );
    }

    private void createKNNVectorDocument(Directory directory) throws IOException {
        IndexWriterConfig conf = newIndexWriterConfig(new MockAnalyzer(random()));
        IndexWriter writer = new IndexWriter(directory, conf);
        Document knnDocument = new Document();
        knnDocument.add(
            new BinaryDocValuesField(
                MOCK_INDEX_FIELD_NAME,
                new VectorField(MOCK_INDEX_FIELD_NAME, SAMPLE_VECTOR_DATA, new FieldType()).binaryValue()
            )
        );
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

    public void testGetValue() throws IOException {
        scriptDocValues.setNextDocId(0);
        Assert.assertArrayEquals(SAMPLE_VECTOR_DATA, scriptDocValues.getValue(), 0.1f);
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
}

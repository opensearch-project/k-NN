/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.KNNTestCase;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.store.Directory;
import org.opensearch.index.fielddata.ScriptDocValues;
import org.junit.Before;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfFloatsSerializer;

import java.io.IOException;

public class KNNVectorDVLeafFieldDataTests extends KNNTestCase {

    private static final String MOCK_INDEX_FIELD_NAME = "test-index-field-name";
    private static final String MOCK_NUMERIC_INDEX_FIELD_NAME = "test-index-price";
    private LeafReaderContext leafReaderContext;
    private Directory directory;
    private DirectoryReader reader;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        directory = newDirectory();
        createKNNVectorDocument(directory);
        reader = DirectoryReader.open(directory);
        leafReaderContext = reader.getContext().leaves().get(0);
    }

    private void createKNNVectorDocument(Directory directory) throws IOException {
        IndexWriterConfig conf = newIndexWriterConfig(new MockAnalyzer(random()));
        IndexWriter writer = new IndexWriter(directory, conf);
        Document knnDocument = new Document();
        byte[] vectorBinary = KNNVectorAsCollectionOfFloatsSerializer.INSTANCE.floatToByteArray(new float[] { 1.0f, 2.0f });
        knnDocument.add(new BinaryDocValuesField(MOCK_INDEX_FIELD_NAME, new BytesRef(vectorBinary)));
        knnDocument.add(new NumericDocValuesField(MOCK_NUMERIC_INDEX_FIELD_NAME, 1000));
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

    public void testGetScriptValues() {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
            leafReaderContext.reader(),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        );
        ScriptDocValues<float[]> scriptValues = leafFieldData.getScriptValues();
        assertNotNull(scriptValues);
        assertTrue(scriptValues instanceof KNNVectorScriptDocValues);
    }

    public void testGetScriptValuesWrongFieldName() {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(leafReaderContext.reader(), "invalid", VectorDataType.FLOAT);
        ScriptDocValues<float[]> scriptValues = leafFieldData.getScriptValues();
        assertNotNull(scriptValues);
    }

    public void testGetScriptValuesWrongFieldType() {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
            leafReaderContext.reader(),
            MOCK_NUMERIC_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        );
        expectThrows(IllegalStateException.class, () -> leafFieldData.getScriptValues());
    }

    public void testRamBytesUsed() {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(leafReaderContext.reader(), "", VectorDataType.FLOAT);
        assertEquals(0, leafFieldData.ramBytesUsed());
    }

    public void testGetBytesValues() {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(leafReaderContext.reader(), "", VectorDataType.FLOAT);
        expectThrows(UnsupportedOperationException.class, () -> leafFieldData.getBytesValues());
    }
}

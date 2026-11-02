/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.opensearch.knn.KNNTestCase;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.store.Directory;
import org.opensearch.index.fielddata.ScriptDocValues;
import org.junit.Before;

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
        knnDocument.add(
            new BinaryDocValuesField(
                MOCK_INDEX_FIELD_NAME,
                new VectorField(MOCK_INDEX_FIELD_NAME, new float[] { 1.0f, 2.0f }, new FieldType()).binaryValue()
            )
        );
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

    public void testScriptValuesUseWrappedFieldView() throws IOException {
        LeafReader filteredReader = new FilterLeafReader(leafReaderContext.reader()) {
            @Override
            public FieldInfos getFieldInfos() {
                return new FieldInfos(new FieldInfo[0]);
            }

            @Override
            public CacheHelper getCoreCacheHelper() {
                return null;
            }

            @Override
            public CacheHelper getReaderCacheHelper() {
                return null;
            }
        };
        KNNVectorScriptDocValues<?> visible = (KNNVectorScriptDocValues<?>) new KNNVectorDVLeafFieldData(
            leafReaderContext.reader(),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        ).getScriptValues();
        KNNVectorScriptDocValues<?> absent = (KNNVectorScriptDocValues<?>) new KNNVectorDVLeafFieldData(
            filteredReader,
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        ).getScriptValues();
        visible.setNextDocId(0);
        absent.setNextDocId(0);
        assertEquals(1, visible.size());
        assertEquals(0, absent.size());
    }

    @SuppressWarnings("unchecked")
    public void testGetScriptValues() {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
            leafReaderContext.reader(),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        );
        ScriptDocValues<float[]> scriptValues = (ScriptDocValues<float[]>) leafFieldData.getScriptValues();
        assertNotNull(scriptValues);
        assertTrue(scriptValues instanceof KNNVectorScriptDocValues);
    }

    @SuppressWarnings("unchecked")
    public void testGetScriptValuesWrongFieldName() {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(leafReaderContext.reader(), "invalid", VectorDataType.FLOAT);
        ScriptDocValues<float[]> scriptValues = (ScriptDocValues<float[]>) leafFieldData.getScriptValues();
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

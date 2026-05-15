/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.opensearch.index.fielddata.ScriptDocValues;
import org.opensearch.index.mapper.DocValueFetcher;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.search.DocValueFormat;
import org.junit.Before;

import java.io.IOException;

public class KNNVectorDVLeafFieldDataTests extends KNNTestCase {

    private static final String MOCK_INDEX_FIELD_NAME = "test-index-field-name";
    private static final String MOCK_NUMERIC_INDEX_FIELD_NAME = "test-index-price";
    private static final float[] SAMPLE_VECTOR_1 = new float[] { 1.0f, 2.0f };
    private static final float[] SAMPLE_VECTOR_2 = new float[] { 3.0f, 4.0f };
    private static final float[] SAMPLE_VECTOR_3 = new float[] { 5.0f, 6.0f };
    private static final float[] SAMPLE_VECTOR_4 = new float[] { 7.0f, 8.0f };
    private static final float[][] ALL_VECTORS = { SAMPLE_VECTOR_1, SAMPLE_VECTOR_2, SAMPLE_VECTOR_3, SAMPLE_VECTOR_4 };
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
        for (int i = 0; i < ALL_VECTORS.length; i++) {
            Document doc = new Document();
            doc.add(new KnnFloatVectorField(MOCK_INDEX_FIELD_NAME, ALL_VECTORS[i], VectorSimilarityFunction.EUCLIDEAN));
            doc.add(new NumericDocValuesField(MOCK_NUMERIC_INDEX_FIELD_NAME, 1000 + i));
            writer.addDocument(doc);
        }
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

    public void testRamBytesUsed() {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
            leafReaderContext.reader(),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        );
        assertEquals(0, leafFieldData.ramBytesUsed());
    }

    public void testGetBytesValues() {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
            leafReaderContext.reader(),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        );
        expectThrows(UnsupportedOperationException.class, () -> leafFieldData.getBytesValues());
    }

    public void testGetLeafValueFetcher_floatVector_returnsCorrectValues() throws IOException {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
            leafReaderContext.reader(),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        );
        DocValueFetcher.Leaf leaf = leafFieldData.getLeafValueFetcher(DocValueFormat.RAW);
        assertNotNull(leaf);

        float[][] results = new float[ALL_VECTORS.length][];
        for (int docId = 0; docId < ALL_VECTORS.length; docId++) {
            assertTrue("advanceExact should succeed for doc " + docId, leaf.advanceExact(docId));
            assertEquals(1, leaf.docValueCount());
            Object value = leaf.nextValue();
            assertTrue(value instanceof float[]);
            results[docId] = (float[]) value;
        }

        for (int docId = 0; docId < ALL_VECTORS.length; docId++) {
            assertArrayEquals("Vector mismatch for doc " + docId, ALL_VECTORS[docId], results[docId], 0.001f);
        }
    }

    public void testGetLeafValueFetcher_advanceExact_nonExistentDoc_returnsFalse() throws IOException {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
            leafReaderContext.reader(),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        );
        DocValueFetcher.Leaf leaf = leafFieldData.getLeafValueFetcher(DocValueFormat.RAW);

        boolean[] advanceResults = new boolean[ALL_VECTORS.length + 1];
        int[] docValueCounts = new int[ALL_VECTORS.length + 1];

        for (int docId = 0; docId < ALL_VECTORS.length; docId++) {
            advanceResults[docId] = leaf.advanceExact(docId);
            docValueCounts[docId] = leaf.docValueCount();
        }
        advanceResults[ALL_VECTORS.length] = leaf.advanceExact(999);
        docValueCounts[ALL_VECTORS.length] = leaf.docValueCount();

        for (int docId = 0; docId < ALL_VECTORS.length; docId++) {
            assertTrue("advanceExact should succeed for doc " + docId, advanceResults[docId]);
            assertEquals("docValueCount should be 1 for doc " + docId, 1, docValueCounts[docId]);
        }
        assertFalse("advanceExact should fail for non-existent doc", advanceResults[ALL_VECTORS.length]);
        assertEquals(0, docValueCounts[ALL_VECTORS.length]);
    }

    public void testGetLeafValueFetcher_docValueCount_isOne() throws IOException {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
            leafReaderContext.reader(),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        );
        DocValueFetcher.Leaf leaf = leafFieldData.getLeafValueFetcher(DocValueFormat.RAW);

        int[] docValueCounts = new int[ALL_VECTORS.length];
        for (int docId = 0; docId < ALL_VECTORS.length; docId++) {
            assertTrue(leaf.advanceExact(docId));
            docValueCounts[docId] = leaf.docValueCount();
        }

        for (int docId = 0; docId < ALL_VECTORS.length; docId++) {
            assertEquals("docValueCount should be 1 for doc " + docId, 1, docValueCounts[docId]);
        }
    }

    public void testGetLeafValueFetcher_multipleDocuments_iteratesCorrectly() throws IOException {
        float[] vector1 = new float[] { 1.0f, 2.0f };
        float[] vector2 = new float[] { 3.0f, 4.0f };
        float[] vector3 = new float[] { 5.0f, 6.0f };

        try (Directory multiDocDir = newDirectory()) {
            IndexWriterConfig conf = newIndexWriterConfig(new MockAnalyzer(random()));
            try (IndexWriter writer = new IndexWriter(multiDocDir, conf)) {

                Document doc1 = new Document();
                doc1.add(new KnnFloatVectorField(MOCK_INDEX_FIELD_NAME, vector1, VectorSimilarityFunction.EUCLIDEAN));
                writer.addDocument(doc1);

                Document doc2 = new Document();
                doc2.add(new KnnFloatVectorField(MOCK_INDEX_FIELD_NAME, vector2, VectorSimilarityFunction.EUCLIDEAN));
                writer.addDocument(doc2);

                Document doc3 = new Document();
                doc3.add(new KnnFloatVectorField(MOCK_INDEX_FIELD_NAME, vector3, VectorSimilarityFunction.EUCLIDEAN));
                writer.addDocument(doc3);

                writer.commit();
            }

            try (DirectoryReader multiDocReader = DirectoryReader.open(multiDocDir)) {
                LeafReaderContext ctx = multiDocReader.getContext().leaves().get(0);

                KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
                    ctx.reader(),
                    MOCK_INDEX_FIELD_NAME,
                    VectorDataType.FLOAT
                );
                DocValueFetcher.Leaf leaf = leafFieldData.getLeafValueFetcher(DocValueFormat.RAW);

                float[][] expected = { vector1, vector2, vector3 };
                float[][] results = new float[expected.length][];
                for (int docId = 0; docId < expected.length; docId++) {
                    assertTrue("advanceExact should succeed for doc " + docId, leaf.advanceExact(docId));
                    assertEquals(1, leaf.docValueCount());
                    results[docId] = (float[]) leaf.nextValue();
                }

                for (int docId = 0; docId < expected.length; docId++) {
                    assertArrayEquals("Vector mismatch for doc " + docId, expected[docId], results[docId], 0.001f);
                }
            }
        }
    }

    public void testGetLeafValueFetcher_byteVectorDataType_throwsUnsupportedOp() {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
            leafReaderContext.reader(),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.BYTE
        );
        UnsupportedOperationException ex = expectThrows(
            UnsupportedOperationException.class,
            () -> leafFieldData.getLeafValueFetcher(DocValueFormat.RAW)
        );
        assertTrue(ex.getMessage().contains("docvalue_fields is not supported"));
        assertTrue(ex.getMessage().contains("BYTE"));
    }

    public void testGetLeafValueFetcher_binaryVectorDataType_throwsUnsupportedOp() {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
            leafReaderContext.reader(),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.BINARY
        );
        UnsupportedOperationException ex = expectThrows(
            UnsupportedOperationException.class,
            () -> leafFieldData.getLeafValueFetcher(DocValueFormat.RAW)
        );
        assertTrue(ex.getMessage().contains("docvalue_fields is not supported"));
        assertTrue(ex.getMessage().contains("BINARY"));
    }

    public void testGetLeafValueFetcher_fieldNotInSegment_returnsEmptyLeaf() throws IOException {
        int numDocs = 3;
        try (Directory noVectorDir = newDirectory()) {
            IndexWriterConfig conf = newIndexWriterConfig(new MockAnalyzer(random()));
            try (IndexWriter writer = new IndexWriter(noVectorDir, conf)) {
                for (int i = 0; i < numDocs; i++) {
                    Document doc = new Document();
                    doc.add(new NumericDocValuesField("other_field", 42 + i));
                    writer.addDocument(doc);
                }
                writer.commit();
            }

            try (DirectoryReader noVectorReader = DirectoryReader.open(noVectorDir)) {
                LeafReaderContext ctx = noVectorReader.getContext().leaves().get(0);

                KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
                    ctx.reader(),
                    MOCK_INDEX_FIELD_NAME,
                    VectorDataType.FLOAT
                );
                DocValueFetcher.Leaf leaf = leafFieldData.getLeafValueFetcher(DocValueFormat.RAW);
                assertNotNull(leaf);

                boolean[] advanceResults = new boolean[numDocs];
                int[] docValueCounts = new int[numDocs];
                for (int docId = 0; docId < numDocs; docId++) {
                    advanceResults[docId] = leaf.advanceExact(docId);
                    docValueCounts[docId] = leaf.docValueCount();
                }

                for (int docId = 0; docId < numDocs; docId++) {
                    assertFalse("advanceExact should fail for doc " + docId + " in segment without vector field", advanceResults[docId]);
                    assertEquals(0, docValueCounts[docId]);
                }
            }
        }
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnByteVectorField;
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
import org.junit.Before;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

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
        DocValueFetcher.Leaf leaf = leafFieldData.getLeafValueFetcher(KNNVectorDocValueFormat.ARRAY_FORMAT);
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
        DocValueFetcher.Leaf leaf = leafFieldData.getLeafValueFetcher(KNNVectorDocValueFormat.ARRAY_FORMAT);

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
        DocValueFetcher.Leaf leaf = leafFieldData.getLeafValueFetcher(KNNVectorDocValueFormat.ARRAY_FORMAT);

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
                DocValueFetcher.Leaf leaf = leafFieldData.getLeafValueFetcher(KNNVectorDocValueFormat.ARRAY_FORMAT);

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

    public void testGetLeafValueFetcher_binaryFormat_returnsByteArray() throws IOException {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
            leafReaderContext.reader(),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        );
        DocValueFetcher.Leaf leaf = leafFieldData.getLeafValueFetcher(KNNVectorDocValueFormat.BINARY_FORMAT);
        assertNotNull(leaf);

        assertTrue(leaf.advanceExact(0));
        assertEquals(1, leaf.docValueCount());
        Object value = leaf.nextValue();
        // Returns byte[] (little-endian floats) which XContentBuilder will base64-encode during serialization
        assertTrue("Binary format for float vector should return byte[]", value instanceof byte[]);
        byte[] bytes = (byte[]) value;
        assertEquals("Byte array length should be dimension * 4", SAMPLE_VECTOR_1.length * Float.BYTES, bytes.length);
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < SAMPLE_VECTOR_1.length; i++) {
            assertEquals("Value mismatch at index " + i, SAMPLE_VECTOR_1[i], buffer.getFloat(), 0.001f);
        }
    }

    public void testGetLeafValueFetcher_nonKNNFormat_throwsIllegalArgument() {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
            leafReaderContext.reader(),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        );
        IllegalArgumentException ex = expectThrows(
            IllegalArgumentException.class,
            () -> leafFieldData.getLeafValueFetcher(org.opensearch.search.DocValueFormat.RAW)
        );
        assertTrue("Error should mention unsupported format", ex.getMessage().contains("Unsupported DocValueFormat"));
        assertTrue("Error should mention the field name", ex.getMessage().contains(MOCK_INDEX_FIELD_NAME));
    }

    public void testGetLeafValueFetcher_nullFormat_throwsIllegalArgument() {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
            leafReaderContext.reader(),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        );
        IllegalArgumentException ex = expectThrows(IllegalArgumentException.class, () -> leafFieldData.getLeafValueFetcher(null));
        assertTrue("Error should mention unsupported format", ex.getMessage().contains("Unsupported DocValueFormat"));
        assertTrue("Error should mention the field name", ex.getMessage().contains(MOCK_INDEX_FIELD_NAME));
    }

    public void testGetLeafValueFetcher_binaryFormat_multipleDocuments() throws IOException {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
            leafReaderContext.reader(),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        );
        DocValueFetcher.Leaf leaf = leafFieldData.getLeafValueFetcher(KNNVectorDocValueFormat.BINARY_FORMAT);

        for (int docId = 0; docId < ALL_VECTORS.length; docId++) {
            assertTrue("advanceExact should succeed for doc " + docId, leaf.advanceExact(docId));
            assertEquals("docValueCount should be 1 for doc " + docId, 1, leaf.docValueCount());
            Object value = leaf.nextValue();
            assertTrue("Binary format should produce a byte[] for doc " + docId, value instanceof byte[]);

            byte[] bytes = (byte[]) value;
            assertEquals("Byte array length mismatch for doc " + docId, ALL_VECTORS[docId].length * Float.BYTES, bytes.length);
            ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < ALL_VECTORS[docId].length; i++) {
                assertEquals("Value mismatch at index " + i + " for doc " + docId, ALL_VECTORS[docId][i], buffer.getFloat(), 0.001f);
            }
        }
    }

    public void testGetLeafValueFetcher_byteVector_arrayFormat_returnsCorrectValues() throws IOException {
        byte[] byteVector1 = new byte[] { 1, -3, 127, -128 };
        byte[] byteVector2 = new byte[] { 0, 50, -50, 100 };

        try (Directory byteVectorDir = newDirectory()) {
            IndexWriterConfig conf = newIndexWriterConfig(new MockAnalyzer(random()));
            try (IndexWriter writer = new IndexWriter(byteVectorDir, conf)) {
                Document doc1 = new Document();
                doc1.add(new KnnByteVectorField(MOCK_INDEX_FIELD_NAME, byteVector1, VectorSimilarityFunction.EUCLIDEAN));
                writer.addDocument(doc1);

                Document doc2 = new Document();
                doc2.add(new KnnByteVectorField(MOCK_INDEX_FIELD_NAME, byteVector2, VectorSimilarityFunction.EUCLIDEAN));
                writer.addDocument(doc2);
                writer.commit();
            }

            try (DirectoryReader byteReader = DirectoryReader.open(byteVectorDir)) {
                LeafReaderContext ctx = byteReader.getContext().leaves().get(0);
                KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
                    ctx.reader(),
                    MOCK_INDEX_FIELD_NAME,
                    VectorDataType.BYTE
                );
                DocValueFetcher.Leaf leaf = leafFieldData.getLeafValueFetcher(KNNVectorDocValueFormat.ARRAY_FORMAT);

                assertTrue("advanceExact should succeed for doc 0", leaf.advanceExact(0));
                assertEquals(1, leaf.docValueCount());
                Object value0 = leaf.nextValue();
                assertTrue("Array format for byte vector should return int[]", value0 instanceof int[]);
                assertArrayEquals("Byte vector mismatch for doc 0", new int[] { 1, -3, 127, -128 }, (int[]) value0);

                assertTrue("advanceExact should succeed for doc 1", leaf.advanceExact(1));
                assertEquals(1, leaf.docValueCount());
                Object value1 = leaf.nextValue();
                assertTrue("Array format for byte vector should return int[]", value1 instanceof int[]);
                assertArrayEquals("Byte vector mismatch for doc 1", new int[] { 0, 50, -50, 100 }, (int[]) value1);
            }
        }
    }

    public void testGetLeafValueFetcher_byteVector_binaryFormat_returnsByteArray() throws IOException {
        byte[] byteVector = new byte[] { 10, 20, -30, 40 };

        try (Directory byteVectorDir = newDirectory()) {
            IndexWriterConfig conf = newIndexWriterConfig(new MockAnalyzer(random()));
            try (IndexWriter writer = new IndexWriter(byteVectorDir, conf)) {
                Document doc = new Document();
                doc.add(new KnnByteVectorField(MOCK_INDEX_FIELD_NAME, byteVector, VectorSimilarityFunction.EUCLIDEAN));
                writer.addDocument(doc);
                writer.commit();
            }

            try (DirectoryReader byteReader = DirectoryReader.open(byteVectorDir)) {
                LeafReaderContext ctx = byteReader.getContext().leaves().get(0);
                KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
                    ctx.reader(),
                    MOCK_INDEX_FIELD_NAME,
                    VectorDataType.BYTE
                );
                DocValueFetcher.Leaf leaf = leafFieldData.getLeafValueFetcher(KNNVectorDocValueFormat.BINARY_FORMAT);

                assertTrue("advanceExact should succeed for doc 0", leaf.advanceExact(0));
                assertEquals(1, leaf.docValueCount());
                Object value = leaf.nextValue();
                // Returns byte[] which XContentBuilder will base64-encode during serialization
                assertTrue("Binary format for byte vector should return byte[]", value instanceof byte[]);
                assertArrayEquals("Byte vector should match original", byteVector, (byte[]) value);
            }
        }
    }

    public void testGetLeafValueFetcher_binaryVector_arrayFormat_returnsCorrectValues() throws IOException {
        // Binary vectors: dimension is in bits, stored as byte[dimension/8]
        byte[] binaryVector1 = new byte[] { (byte) 0b10101010, (byte) 0b01010101 }; // 16 bits
        byte[] binaryVector2 = new byte[] { (byte) 0xFF, (byte) 0x00 };

        try (Directory binaryVectorDir = newDirectory()) {
            IndexWriterConfig conf = newIndexWriterConfig(new MockAnalyzer(random()));
            try (IndexWriter writer = new IndexWriter(binaryVectorDir, conf)) {
                Document doc1 = new Document();
                doc1.add(new KnnByteVectorField(MOCK_INDEX_FIELD_NAME, binaryVector1, VectorSimilarityFunction.EUCLIDEAN));
                writer.addDocument(doc1);

                Document doc2 = new Document();
                doc2.add(new KnnByteVectorField(MOCK_INDEX_FIELD_NAME, binaryVector2, VectorSimilarityFunction.EUCLIDEAN));
                writer.addDocument(doc2);
                writer.commit();
            }

            try (DirectoryReader binaryReader = DirectoryReader.open(binaryVectorDir)) {
                LeafReaderContext ctx = binaryReader.getContext().leaves().get(0);
                KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
                    ctx.reader(),
                    MOCK_INDEX_FIELD_NAME,
                    VectorDataType.BINARY
                );
                DocValueFetcher.Leaf leaf = leafFieldData.getLeafValueFetcher(KNNVectorDocValueFormat.ARRAY_FORMAT);

                assertTrue("advanceExact should succeed for doc 0", leaf.advanceExact(0));
                assertEquals(1, leaf.docValueCount());
                Object value0 = leaf.nextValue();
                assertTrue("Array format for binary vector should return int[]", value0 instanceof int[]);
                int[] expected0 = new int[] { (byte) 0b10101010, (byte) 0b01010101 };
                assertArrayEquals("Binary vector mismatch for doc 0", expected0, (int[]) value0);

                assertTrue("advanceExact should succeed for doc 1", leaf.advanceExact(1));
                assertEquals(1, leaf.docValueCount());
                Object value1 = leaf.nextValue();
                assertTrue("Array format for binary vector should return int[]", value1 instanceof int[]);
                int[] expected1 = new int[] { (byte) 0xFF, (byte) 0x00 };
                assertArrayEquals("Binary vector mismatch for doc 1", expected1, (int[]) value1);
            }
        }
    }

    public void testGetLeafValueFetcher_binaryVector_binaryFormat_returnsByteArray() throws IOException {
        byte[] binaryVector = new byte[] { (byte) 0xAB, (byte) 0xCD, (byte) 0xEF, (byte) 0x01 };

        try (Directory binaryVectorDir = newDirectory()) {
            IndexWriterConfig conf = newIndexWriterConfig(new MockAnalyzer(random()));
            try (IndexWriter writer = new IndexWriter(binaryVectorDir, conf)) {
                Document doc = new Document();
                doc.add(new KnnByteVectorField(MOCK_INDEX_FIELD_NAME, binaryVector, VectorSimilarityFunction.EUCLIDEAN));
                writer.addDocument(doc);
                writer.commit();
            }

            try (DirectoryReader binaryReader = DirectoryReader.open(binaryVectorDir)) {
                LeafReaderContext ctx = binaryReader.getContext().leaves().get(0);
                KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
                    ctx.reader(),
                    MOCK_INDEX_FIELD_NAME,
                    VectorDataType.BINARY
                );
                DocValueFetcher.Leaf leaf = leafFieldData.getLeafValueFetcher(KNNVectorDocValueFormat.BINARY_FORMAT);

                assertTrue("advanceExact should succeed for doc 0", leaf.advanceExact(0));
                assertEquals(1, leaf.docValueCount());
                Object value = leaf.nextValue();
                // Returns byte[] which XContentBuilder will base64-encode during serialization
                assertTrue("Binary format for binary vector should return byte[]", value instanceof byte[]);
                assertArrayEquals("Binary vector should match original", binaryVector, (byte[]) value);
            }
        }
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
                DocValueFetcher.Leaf leaf = leafFieldData.getLeafValueFetcher(KNNVectorDocValueFormat.ARRAY_FORMAT);
                assertNotNull("Empty leaf should not be null", leaf);

                // Verify advanceExact returns false for all docs
                for (int docId = 0; docId < numDocs; docId++) {
                    assertFalse("advanceExact should fail for doc " + docId + " in segment without vector field", leaf.advanceExact(docId));
                    assertEquals("docValueCount should be 0 for doc " + docId, 0, leaf.docValueCount());
                }

                // Verify nextValue returns null on the empty leaf
                assertNull("Empty leaf nextValue should return null", leaf.nextValue());
            }
        }
    }
}

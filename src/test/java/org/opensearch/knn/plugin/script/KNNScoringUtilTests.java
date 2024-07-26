/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import java.util.Arrays;
import java.util.Locale;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNVectorScriptDocValues;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.VectorField;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.store.Directory;

import java.io.IOException;
import java.math.BigInteger;
import java.util.List;
import java.util.function.BiFunction;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class KNNScoringUtilTests extends KNNTestCase {

    private List<Number> getTestQueryVector() {
        return List.of(1.0f, 1.0f, 1.0f);
    }

    private List<Number> getTestZeroVector() {
        return List.of(0.0f, 0.0f, 0.0f);
    }

    public void testL2SquaredScoringFunction() {
        float[] queryVector = { 1.0f, 1.0f, 1.0f };
        float[] inputVector = { 4.0f, 4.0f, 4.0f };

        Float distance = KNNScoringUtil.l2Squared(queryVector, inputVector);
        assertTrue(distance == 27.0f);
    }

    public void testWrongDimensionL2SquaredScoringFunction() {
        float[] queryVector = { 1.0f, 1.0f };
        float[] inputVector = { 4.0f, 4.0f, 4.0f };
        expectThrows(IllegalArgumentException.class, () -> KNNScoringUtil.l2Squared(queryVector, inputVector));
    }

    public void testCosineSimilScoringFunction() {
        float[] queryVector = { 1.0f, 1.0f, 1.0f };
        float[] inputVector = { 4.0f, 4.0f, 4.0f };

        float queryVectorMagnitude = KNNScoringSpaceUtil.getVectorMagnitudeSquared(queryVector);
        float inputVectorMagnitude = KNNScoringSpaceUtil.getVectorMagnitudeSquared(inputVector);
        float dotProduct = 12.0f;
        float expectedScore = (float) (dotProduct / (Math.sqrt(queryVectorMagnitude * inputVectorMagnitude)));

        Float actualScore = KNNScoringUtil.cosinesimil(queryVector, inputVector);
        assertEquals(expectedScore, actualScore, 0.0001);
    }

    public void testCosineSimilOptimizedScoringFunction() {
        float[] queryVector = { 1.0f, 1.0f, 1.0f };
        float[] inputVector = { 4.0f, 4.0f, 4.0f };
        float queryVectorMagnitude = KNNScoringSpaceUtil.getVectorMagnitudeSquared(queryVector);
        float inputVectorMagnitude = KNNScoringSpaceUtil.getVectorMagnitudeSquared(inputVector);
        float dotProduct = 12.0f;
        float expectedScore = (float) (dotProduct / (Math.sqrt(queryVectorMagnitude * inputVectorMagnitude)));

        Float actualScore = KNNScoringUtil.cosinesimilOptimized(queryVector, inputVector, queryVectorMagnitude);
        assertEquals(expectedScore, actualScore, 0.0001);
    }

    public void testGetInvalidVectorMagnitudeSquared() {
        float[] queryVector = null;
        // vector cannot be null
        expectThrows(IllegalStateException.class, () -> KNNScoringSpaceUtil.getVectorMagnitudeSquared(queryVector));
    }

    public void testConvertInvalidVectorToPrimitive() {
        float[] primitiveVector = null;
        assertEquals(primitiveVector, KNNScoringSpaceUtil.convertVectorToPrimitive(primitiveVector, VectorDataType.FLOAT));
    }

    public void testCosineSimilQueryVectorZeroMagnitude() {
        float[] queryVector = { 0, 0 };
        float[] inputVector = { 4.0f, 4.0f };
        assertEquals(0, KNNScoringUtil.cosinesimil(queryVector, inputVector), 0.00001);
    }

    public void testCosineSimilOptimizedQueryVectorZeroMagnitude() {
        float[] inputVector = { 4.0f, 4.0f };
        float[] queryVector = { 0, 0 };
        assertTrue(0 == KNNScoringUtil.cosinesimilOptimized(queryVector, inputVector, 0.0f));
    }

    public void testWrongDimensionCosineSimilScoringFunction() {
        float[] queryVector = { 1.0f, 1.0f };
        float[] inputVector = { 4.0f, 4.0f, 4.0f };
        expectThrows(IllegalArgumentException.class, () -> KNNScoringUtil.cosinesimil(queryVector, inputVector));
    }

    public void testWrongDimensionCosineSimilOPtimizedScoringFunction() {
        float[] queryVector = { 1.0f, 1.0f };
        float[] inputVector = { 4.0f, 4.0f, 4.0f };
        expectThrows(IllegalArgumentException.class, () -> KNNScoringUtil.cosinesimilOptimized(queryVector, inputVector, 1.0f));
    }

    public void testBitHammingDistance_BitSet() {
        BigInteger bigInteger1 = new BigInteger("4", 16);
        BigInteger bigInteger2 = new BigInteger("32278", 16);
        BigInteger bigInteger3 = new BigInteger("AB5432", 16);
        BigInteger bigInteger4 = new BigInteger("EECCDDFF", 16);
        BigInteger bigInteger5 = new BigInteger("1114AB5432", 16);

        /*
         * Hex to binary table:
         *
         * 4            -> 0000 0000 0000 0000 0000 0000 0000 0000 0000 0100
         * 32278        -> 0000 0000 0000 0000 0000 0011 0010 0010 0111 1000
         * AB5432       -> 0000 0000 0000 0000 1010 1011 0101 0100 0011 0010
         * EECCDDFF     -> 0000 0000 1110 1110 1100 1100 1101 1101 1111 1111
         * 1114AB5432   -> 0001 0001 0001 0100 1010 1011 0101 0100 0011 0010
         */

        assertEquals(9.0, KNNScoringUtil.calculateHammingBit(bigInteger1, bigInteger2), 0.1);
        assertEquals(12.0, KNNScoringUtil.calculateHammingBit(bigInteger1, bigInteger3), 0.1);
        assertEquals(23.0, KNNScoringUtil.calculateHammingBit(bigInteger1, bigInteger4), 0.1);
        assertEquals(16.0, KNNScoringUtil.calculateHammingBit(bigInteger1, bigInteger5), 0.1);

        assertEquals(9.0, KNNScoringUtil.calculateHammingBit(bigInteger2, bigInteger1), 0.1);
        assertEquals(11.0, KNNScoringUtil.calculateHammingBit(bigInteger2, bigInteger3), 0.1);
        assertEquals(24.0, KNNScoringUtil.calculateHammingBit(bigInteger2, bigInteger4), 0.1);
        assertEquals(15.0, KNNScoringUtil.calculateHammingBit(bigInteger2, bigInteger5), 0.1);

        assertEquals(12.0, KNNScoringUtil.calculateHammingBit(bigInteger3, bigInteger1), 0.1);
        assertEquals(11.0, KNNScoringUtil.calculateHammingBit(bigInteger3, bigInteger2), 0.1);
        assertEquals(19.0, KNNScoringUtil.calculateHammingBit(bigInteger3, bigInteger4), 0.1);
        assertEquals(4.0, KNNScoringUtil.calculateHammingBit(bigInteger3, bigInteger5), 0.1);

        assertEquals(23.0, KNNScoringUtil.calculateHammingBit(bigInteger4, bigInteger1), 0.1);
        assertEquals(24.0, KNNScoringUtil.calculateHammingBit(bigInteger4, bigInteger2), 0.1);
        assertEquals(19.0, KNNScoringUtil.calculateHammingBit(bigInteger4, bigInteger3), 0.1);
        assertEquals(21.0, KNNScoringUtil.calculateHammingBit(bigInteger4, bigInteger5), 0.1);

        assertEquals(16.0, KNNScoringUtil.calculateHammingBit(bigInteger5, bigInteger1), 0.1);
        assertEquals(15.0, KNNScoringUtil.calculateHammingBit(bigInteger5, bigInteger2), 0.1);
        assertEquals(4.0, KNNScoringUtil.calculateHammingBit(bigInteger5, bigInteger3), 0.1);
        assertEquals(21.0, KNNScoringUtil.calculateHammingBit(bigInteger5, bigInteger4), 0.1);
    }

    public void testBitHammingDistance_Long() {
        Long long1 = 1_817L;
        Long long2 = 500_000_924_849_631L;
        Long long3 = -500_000_924_849_631L;

        /*
         * 64 bit 2's complement:
         * 1_817L                  -> 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0111 0001 1001
         * 500_000_924_849_631L    -> 0000 0000 0000 0001 1100 0110 1011 1111 1000 1001 1000 0011 0101 0101 1101 1111
         * -500_000_924_849_631L   -> 1111 1111 1111 1110 0011 1001 0100 0000 0111 0110 0111 1100 1010 1010 0010 0001
         */

        assertEquals(25.0, KNNScoringUtil.calculateHammingBit(long1, long2), 0.1);
        assertEquals(38.0, KNNScoringUtil.calculateHammingBit(long1, long3), 0.1);
        assertEquals(63.0, KNNScoringUtil.calculateHammingBit(long2, long3), 0.1);
        assertEquals(0.0, KNNScoringUtil.calculateHammingBit(long3, long3), 0.1);
    }

    public void testL2SquaredAllowlistedScoringFunction() throws IOException {
        List<Number> queryVector = getTestQueryVector();
        TestKNNScriptDocValues dataset = new TestKNNScriptDocValues();
        dataset.createKNNVectorDocument(new float[] { 4.0f, 4.0f, 4.0f }, "test-index-field-name");
        KNNVectorScriptDocValues scriptDocValues = dataset.getScriptDocValues("test-index-field-name");
        scriptDocValues.setNextDocId(0);
        Float distance = KNNScoringUtil.l2Squared(queryVector, scriptDocValues);
        assertEquals(27.0f, distance, 0.1f);
        dataset.close();
    }

    public void testScriptDocValuesFailsL2() throws IOException {
        List<Number> queryVector = getTestQueryVector();
        TestKNNScriptDocValues dataset = new TestKNNScriptDocValues();
        dataset.createKNNVectorDocument(new float[] { 4.0f, 4.0f, 4.0f }, "test-index-field-name");
        KNNVectorScriptDocValues scriptDocValues = dataset.getScriptDocValues("test-index-field-name");
        expectThrows(IllegalStateException.class, () -> KNNScoringUtil.l2Squared(queryVector, scriptDocValues));
        dataset.close();
    }

    public void testCosineSimilarityScoringFunction() throws IOException {
        List<Number> queryVector = getTestQueryVector();
        TestKNNScriptDocValues dataset = new TestKNNScriptDocValues();
        dataset.createKNNVectorDocument(new float[] { 4.0f, 4.0f, 4.0f }, "test-index-field-name");
        KNNVectorScriptDocValues scriptDocValues = dataset.getScriptDocValues("test-index-field-name");
        scriptDocValues.setNextDocId(0);

        Float actualScore = KNNScoringUtil.cosineSimilarity(queryVector, scriptDocValues);
        assertEquals(1.0f, actualScore, 0.0001);
        dataset.close();
    }

    public void testScriptDocValuesFailsCosineSimilarity() throws IOException {
        List<Number> queryVector = getTestQueryVector();
        TestKNNScriptDocValues dataset = new TestKNNScriptDocValues();
        dataset.createKNNVectorDocument(new float[] { 4.0f, 4.0f, 4.0f }, "test-index-field-name");
        KNNVectorScriptDocValues scriptDocValues = dataset.getScriptDocValues("test-index-field-name");
        expectThrows(IllegalStateException.class, () -> KNNScoringUtil.cosineSimilarity(queryVector, scriptDocValues));
        dataset.close();
    }

    public void testZeroVectorFailsCosineSimilarity() throws IOException {
        List<Number> queryVector = getTestZeroVector();
        TestKNNScriptDocValues dataset = new TestKNNScriptDocValues();
        dataset.createKNNVectorDocument(new float[] { 4.0f, 4.0f, 4.0f }, "test-index-field-name");
        KNNVectorScriptDocValues scriptDocValues = dataset.getScriptDocValues("test-index-field-name");
        scriptDocValues.setNextDocId(0);

        IllegalArgumentException exception = expectThrows(
            IllegalArgumentException.class,
            () -> KNNScoringUtil.cosineSimilarity(queryVector, scriptDocValues)
        );
        assertEquals(
            String.format(Locale.ROOT, "zero vector is not supported when space type is [%s]", SpaceType.COSINESIMIL.getValue()),
            exception.getMessage()
        );
        dataset.close();
    }

    public void testCosineSimilarityOptimizedScoringFunction() throws IOException {
        List<Number> queryVector = getTestQueryVector();
        TestKNNScriptDocValues dataset = new TestKNNScriptDocValues();
        dataset.createKNNVectorDocument(new float[] { 4.0f, 4.0f, 4.0f }, "test-index-field-name");
        KNNVectorScriptDocValues scriptDocValues = dataset.getScriptDocValues("test-index-field-name");
        scriptDocValues.setNextDocId(0);
        Float actualScore = KNNScoringUtil.cosineSimilarity(queryVector, scriptDocValues, 3.0f);
        assertEquals(1.0f, actualScore, 0.0001);
        dataset.close();
    }

    public void testScriptDocValuesFailsCosineSimilarityOptimized() throws IOException {
        List<Number> queryVector = getTestQueryVector();
        TestKNNScriptDocValues dataset = new TestKNNScriptDocValues();
        dataset.createKNNVectorDocument(new float[] { 4.0f, 4.0f, 4.0f }, "test-index-field-name");
        KNNVectorScriptDocValues scriptDocValues = dataset.getScriptDocValues("test-index-field-name");
        dataset.close();
    }

    public void testZeroVectorFailsCosineSimilarityOptimized() throws IOException {
        List<Number> queryVector = getTestZeroVector();
        TestKNNScriptDocValues dataset = new TestKNNScriptDocValues();
        dataset.createKNNVectorDocument(new float[] { 4.0f, 4.0f, 4.0f }, "test-index-field-name");
        KNNVectorScriptDocValues scriptDocValues = dataset.getScriptDocValues("test-index-field-name");
        scriptDocValues.setNextDocId(0);

        IllegalArgumentException exception = expectThrows(
            IllegalArgumentException.class,
            () -> KNNScoringUtil.cosineSimilarity(queryVector, scriptDocValues, 3.0f)
        );
        assertEquals(
            String.format(Locale.ROOT, "zero vector is not supported when space type is [%s]", SpaceType.COSINESIMIL.getValue()),
            exception.getMessage()
        );
        dataset.close();
    }

    public void testCalculateHammingBit_whenByte_thenSuccess() {
        byte[] v1 = { 1, 16, -128 };  // 0000 0001, 0001 0000, 1000 0000
        byte[] v2 = { 2, 17, -1 };    // 0000 0010, 0001 0001, 1111 1111
        assertEquals(10, KNNScoringUtil.calculateHammingBit(v1, v2), 0.001f);
    }

    private void validateThrowExceptionOnGivenDataType(
        final BiFunction<List<Number>, KNNVectorScriptDocValues, Float> func,
        final VectorDataType dataType,
        final String errorMsg
    ) {
        List<Number> queryVector = Arrays.asList(1, 2);
        KNNVectorScriptDocValues docValues = mock(KNNVectorScriptDocValues.class);
        when(docValues.getVectorDataType()).thenReturn(dataType);
        Exception e = expectThrows(IllegalArgumentException.class, () -> func.apply(queryVector, docValues));
        assertTrue(e.getMessage().contains(errorMsg));
    }

    public void testLInfNorm_whenKNNVectorScriptDocValuesOfBinary_thenThrowException() {
        validateThrowExceptionOnGivenDataType(KNNScoringUtil::lInfNorm, VectorDataType.BINARY, "should be either float or byte");
    }

    public void testL1Norm_whenKNNVectorScriptDocValuesOfBinary_thenThrowException() {
        validateThrowExceptionOnGivenDataType(KNNScoringUtil::l1Norm, VectorDataType.BINARY, "should be either float or byte");
    }

    public void testInnerProduct_whenKNNVectorScriptDocValuesOfBinary_thenThrowException() {
        validateThrowExceptionOnGivenDataType(KNNScoringUtil::innerProduct, VectorDataType.BINARY, "should be either float or byte");
    }

    public void testCosineSimilarity_whenKNNVectorScriptDocValuesOfBinary_thenThrowException() {
        validateThrowExceptionOnGivenDataType(KNNScoringUtil::cosineSimilarity, VectorDataType.BINARY, "should be either float or byte");
    }

    public void testHamming_whenKNNVectorScriptDocValuesOfNonBinary_thenThrowException() {
        validateThrowExceptionOnGivenDataType(KNNScoringUtil::hamming, VectorDataType.FLOAT, "should be binary");
    }

    public void testHamming_whenKNNVectorScriptDocValuesOfBinary_thenSuccess() {
        byte[] b1 = { 1, 16, -128 };  // 0000 0001, 0001 0000, 1000 0000
        byte[] b2 = { 2, 17, -1 };    // 0000 0010, 0001 0001, 1111 1111
        float[] f1 = { 1, 16, -128 };  // 0000 0001, 0001 0000, 1000 0000
        float[] f2 = { 2, 17, -1 };    // 0000 0010, 0001 0001, 1111 1111
        List<Number> queryVector = Arrays.asList(f1[0], f1[1], f1[2]);
        KNNVectorScriptDocValues docValues = mock(KNNVectorScriptDocValues.class);
        when(docValues.getVectorDataType()).thenReturn(VectorDataType.BINARY);
        when(docValues.getValue()).thenReturn(f2);
        assertEquals(KNNScoringUtil.calculateHammingBit(b1, b2), KNNScoringUtil.hamming(queryVector, docValues), 0.01f);
    }

    class TestKNNScriptDocValues {
        private KNNVectorScriptDocValues scriptDocValues;
        private Directory directory;
        private DirectoryReader reader;

        TestKNNScriptDocValues() throws IOException {
            directory = newDirectory();
        }

        public KNNVectorScriptDocValues getScriptDocValues(String fieldName) throws IOException {
            if (scriptDocValues == null) {
                reader = DirectoryReader.open(directory);
                LeafReaderContext leafReaderContext = reader.getContext().leaves().get(0);
                scriptDocValues = KNNVectorScriptDocValues.create(
                    leafReaderContext.reader().getBinaryDocValues(fieldName),
                    fieldName,
                    VectorDataType.FLOAT
                );
            }
            return scriptDocValues;
        }

        public void close() throws IOException {
            if (reader != null) reader.close();
            if (directory != null) directory.close();
        }

        public void createKNNVectorDocument(final float[] content, final String fieldName) throws IOException {
            IndexWriterConfig conf = newIndexWriterConfig(new MockAnalyzer(random()));
            IndexWriter writer = new IndexWriter(directory, conf);
            conf.setMergePolicy(NoMergePolicy.INSTANCE); // prevent merges for this test
            Document knnDocument = new Document();
            knnDocument.add(new BinaryDocValuesField(fieldName, new VectorField(fieldName, content, new FieldType()).binaryValue()));
            writer.addDocument(knnDocument);
            writer.commit();
            writer.close();
        }
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.Locale;

import lombok.SneakyThrows;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.index.mapper.BinaryFieldMapper;
import org.opensearch.index.mapper.NumberFieldMapper;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collections;
import java.util.List;
import java.util.function.BiFunction;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.plugin.script.KNNScoringSpace.KNNFieldSpace.DATA_TYPES_DEFAULT;

public class KNNScoringSpaceTests extends KNNTestCase {

    private void expectThrowsExceptionWithNonKNNField(Class clazz) throws NoSuchMethodException {
        Constructor<?> constructor = clazz.getConstructor(Object.class, MappedFieldType.class);
        NumberFieldMapper.NumberFieldType invalidFieldType = mock(NumberFieldMapper.NumberFieldType.class);
        Exception e = expectThrows(InvocationTargetException.class, () -> constructor.newInstance(null, invalidFieldType));
        assertTrue(e.getCause() instanceof IllegalArgumentException);
        assertTrue(e.getCause().getMessage(), e.getCause().getMessage().contains("The field type must be knn_vector"));
    }

    private void expectThrowsExceptionWithKNNFieldWithBinaryDataType(Class clazz) throws NoSuchMethodException {
        Constructor<?> constructor = clazz.getConstructor(Object.class, MappedFieldType.class);
        KNNVectorFieldMapper.KNNVectorFieldType invalidFieldType = mock(KNNVectorFieldMapper.KNNVectorFieldType.class);
        when(invalidFieldType.getVectorDataType()).thenReturn(VectorDataType.BINARY);
        Exception e = expectThrows(InvocationTargetException.class, () -> constructor.newInstance(null, invalidFieldType));
        assertTrue(e.getCause() instanceof IllegalArgumentException);
        assertTrue(
            e.getCause().getMessage(),
            e.getCause().getMessage().contains(String.format("The data type should be %s", DATA_TYPES_DEFAULT))
        );
    }

    @SneakyThrows
    public void testL2_whenValid_thenSucceed() {
        float[] arrayFloat = new float[] { 1.0f, 2.0f, 3.0f };
        List<Double> arrayListQueryObject = new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0));
        KNNMethodContext knnMethodContext = KNNMethodContext.getDefault();
        KNNVectorFieldMapper.KNNVectorFieldType fieldType = new KNNVectorFieldMapper.KNNVectorFieldType(
            "test",
            Collections.emptyMap(),
            3,
            knnMethodContext
        );
        KNNScoringSpace.L2 l2 = new KNNScoringSpace.L2(arrayListQueryObject, fieldType);
        assertEquals(1F, l2.getScoringMethod().apply(arrayFloat, arrayFloat), 0.1F);
    }

    @SneakyThrows
    public void testL2_whenInvalidType_thenException() {
        expectThrowsExceptionWithNonKNNField(KNNScoringSpace.L2.class);
        expectThrowsExceptionWithKNNFieldWithBinaryDataType(KNNScoringSpace.L2.class);
    }

    public void testCosineSimilarity_whenValid_thenSucceed() {
        float[] arrayFloat = new float[] { 1.0f, 2.0f, 3.0f };
        List<Double> arrayListQueryObject = new ArrayList<>(Arrays.asList(2.0, 4.0, 6.0));
        float[] arrayFloat2 = new float[] { 2.0f, 4.0f, 6.0f };
        KNNMethodContext knnMethodContext = KNNMethodContext.getDefault();

        KNNVectorFieldMapper.KNNVectorFieldType fieldType = new KNNVectorFieldMapper.KNNVectorFieldType(
            "test",
            Collections.emptyMap(),
            3,
            knnMethodContext
        );
        KNNScoringSpace.CosineSimilarity cosineSimilarity = new KNNScoringSpace.CosineSimilarity(arrayListQueryObject, fieldType);
        assertEquals(2F, cosineSimilarity.getScoringMethod().apply(arrayFloat2, arrayFloat), 0.1F);

        // invalid zero vector
        final List<Float> queryZeroVector = List.of(0.0f, 0.0f, 0.0f);
        IllegalArgumentException exception1 = expectThrows(
            IllegalArgumentException.class,
            () -> new KNNScoringSpace.CosineSimilarity(queryZeroVector, fieldType)
        );
        assertEquals(
            String.format(Locale.ROOT, "zero vector is not supported when space type is [%s]", SpaceType.COSINESIMIL.getValue()),
            exception1.getMessage()
        );
    }

    public void testCosineSimilarity_whenZeroVector_thenException() {
        KNNMethodContext knnMethodContext = KNNMethodContext.getDefault();
        KNNVectorFieldMapper.KNNVectorFieldType fieldType = new KNNVectorFieldMapper.KNNVectorFieldType(
            "test",
            Collections.emptyMap(),
            3,
            knnMethodContext
        );

        final List<Float> queryZeroVector = List.of(0.0f, 0.0f, 0.0f);
        IllegalArgumentException exception1 = expectThrows(
            IllegalArgumentException.class,
            () -> new KNNScoringSpace.CosineSimilarity(queryZeroVector, fieldType)
        );
        assertEquals(
            String.format(Locale.ROOT, "zero vector is not supported when space type is [%s]", SpaceType.COSINESIMIL.getValue()),
            exception1.getMessage()
        );
    }

    @SneakyThrows
    public void testCosineSimilarity_whenInvalidType_thenException() {
        expectThrowsExceptionWithNonKNNField(KNNScoringSpace.CosineSimilarity.class);
        expectThrowsExceptionWithKNNFieldWithBinaryDataType(KNNScoringSpace.CosineSimilarity.class);
    }

    public void testInnerProd_whenValid_thenSucceed() {
        float[] arrayFloat_case1 = new float[] { 1.0f, 2.0f, 3.0f };
        List<Double> arrayListQueryObject_case1 = new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0));
        float[] arrayFloat2_case1 = new float[] { 1.0f, 1.0f, 1.0f };
        KNNMethodContext knnMethodContext = KNNMethodContext.getDefault();

        KNNVectorFieldMapper.KNNVectorFieldType fieldType = new KNNVectorFieldMapper.KNNVectorFieldType(
            "test",
            Collections.emptyMap(),
            3,
            knnMethodContext
        );
        KNNScoringSpace.InnerProd innerProd = new KNNScoringSpace.InnerProd(arrayListQueryObject_case1, fieldType);

        assertEquals(7.0F, innerProd.getScoringMethod().apply(arrayFloat_case1, arrayFloat2_case1), 0.001F);

        float[] arrayFloat_case2 = new float[] { 100_000.0f, 200_000.0f, 300_000.0f };
        List<Double> arrayListQueryObject_case2 = new ArrayList<>(Arrays.asList(100_000.0, 200_000.0, 300_000.0));
        float[] arrayFloat2_case2 = new float[] { -100_000.0f, -200_000.0f, -300_000.0f };

        innerProd = new KNNScoringSpace.InnerProd(arrayListQueryObject_case2, fieldType);

        assertEquals(7.142857143E-12F, innerProd.getScoringMethod().apply(arrayFloat_case2, arrayFloat2_case2), 1.0E-11F);

        float[] arrayFloat_case3 = new float[] { 100_000.0f, 200_000.0f, 300_000.0f };
        List<Double> arrayListQueryObject_case3 = new ArrayList<>(Arrays.asList(100_000.0, 200_000.0, 300_000.0));
        float[] arrayFloat2_case3 = new float[] { 100_000.0f, 200_000.0f, 300_000.0f };

        innerProd = new KNNScoringSpace.InnerProd(arrayListQueryObject_case3, fieldType);

        assertEquals(140_000_000_001F, innerProd.getScoringMethod().apply(arrayFloat_case3, arrayFloat2_case3), 0.01F);
    }

    @SneakyThrows
    public void testInnerProd_whenInvalidType_thenException() {
        expectThrowsExceptionWithNonKNNField(KNNScoringSpace.InnerProd.class);
        expectThrowsExceptionWithKNNFieldWithBinaryDataType(KNNScoringSpace.InnerProd.class);
    }

    @SuppressWarnings("unchecked")
    public void testHammingBit_Long() {
        NumberFieldMapper.NumberFieldType fieldType = new NumberFieldMapper.NumberFieldType("field", NumberFieldMapper.NumberType.LONG);
        Long longObject1 = 1234L; // ..._0000_0100_1101_0010
        Long longObject2 = 2468L; // ..._0000_1001_1010_0100
        KNNScoringSpace.HammingBit hammingBit = new KNNScoringSpace.HammingBit(longObject1, fieldType);

        assertEquals(0.1111F, ((BiFunction<Long, Long, Float>) hammingBit.scoringMethod).apply(longObject1, longObject2), 0.1F);
    }

    @SuppressWarnings("unchecked")
    public void testHammingBit_Base64() {
        BinaryFieldMapper.BinaryFieldType fieldType = new BinaryFieldMapper.BinaryFieldType("field");
        String base64Object1 = "q83vQUI=";
        String base64Object2 = "//43ITI=";

        /*
         * Base64 to Binary
         * q83vQUI= -> 1010 1011 1100 1101 1110 1111 0100 0001 0100 0010
         * //43ITI= -> 1111 1111 1111 1110 0011 0111 0010 0001 0011 0010
         */

        float expectedResult = 1F / (1 + 16);
        KNNScoringSpace.HammingBit hammingBit = new KNNScoringSpace.HammingBit(base64Object1, fieldType);

        assertEquals(
            expectedResult,
            ((BiFunction<BigInteger, BigInteger, Float>) hammingBit.scoringMethod).apply(
                new BigInteger(Base64.getDecoder().decode(base64Object1)),
                new BigInteger(Base64.getDecoder().decode(base64Object2))
            ),
            0.1F
        );
    }

    public void testHamming_whenKNNFieldType_thenSucceed() {
        List<Double> arrayListQueryObject = new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0));
        KNNMethodContext knnMethodContext = KNNMethodContext.getDefault();
        KNNVectorFieldMapper.KNNVectorFieldType fieldType = new KNNVectorFieldMapper.KNNVectorFieldType(
            "test",
            Collections.emptyMap(),
            8 * arrayListQueryObject.size(),
            knnMethodContext,
            VectorDataType.BINARY
        );
        KNNScoringSpace.Hamming hamming = new KNNScoringSpace.Hamming(arrayListQueryObject, fieldType);

        float[] arrayFloat = new float[] { 1.0f, 2.0f, 3.0f };
        assertEquals(1F, hamming.getScoringMethod().apply(arrayFloat, arrayFloat), 0.1F);
    }

    public void testHamming_whenNonBinaryVectorDataType_thenException() {
        KNNVectorFieldMapper.KNNVectorFieldType invalidFieldType = mock(KNNVectorFieldMapper.KNNVectorFieldType.class);
        when(invalidFieldType.getVectorDataType()).thenReturn(randomInt() % 2 == 0 ? VectorDataType.FLOAT : VectorDataType.BYTE);
        Exception e = expectThrows(IllegalArgumentException.class, () -> new KNNScoringSpace.Hamming(null, invalidFieldType));
        assertTrue(e.getMessage(), e.getMessage().contains("The data type should be [BINARY]"));
    }
}

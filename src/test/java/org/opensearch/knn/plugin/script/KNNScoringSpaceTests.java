/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import java.util.Locale;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.SpaceType;
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

public class KNNScoringSpaceTests extends KNNTestCase {

    public void testL2() {
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
        assertEquals(1F, l2.scoringMethod.apply(arrayFloat, arrayFloat), 0.1F);

        NumberFieldMapper.NumberFieldType invalidFieldType = new NumberFieldMapper.NumberFieldType(
            "field",
            NumberFieldMapper.NumberType.INTEGER
        );
        expectThrows(IllegalArgumentException.class, () -> new KNNScoringSpace.L2(arrayListQueryObject, invalidFieldType));
    }

    public void testCosineSimilarity() {
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

        assertEquals(2F, cosineSimilarity.scoringMethod.apply(arrayFloat2, arrayFloat), 0.1F);

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

        NumberFieldMapper.NumberFieldType invalidFieldType = new NumberFieldMapper.NumberFieldType(
            "field",
            NumberFieldMapper.NumberType.INTEGER
        );
        IllegalArgumentException exception2 = expectThrows(
            IllegalArgumentException.class,
            () -> new KNNScoringSpace.CosineSimilarity(arrayListQueryObject, invalidFieldType)
        );
        assertEquals("Incompatible field_type for cosine space. The field type must be knn_vector.", exception2.getMessage());
    }

    public void testInnerProdSimilarity() {
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

        assertEquals(7.0F, innerProd.scoringMethod.apply(arrayFloat_case1, arrayFloat2_case1), 0.001F);

        float[] arrayFloat_case2 = new float[] { 100_000.0f, 200_000.0f, 300_000.0f };
        List<Double> arrayListQueryObject_case2 = new ArrayList<>(Arrays.asList(100_000.0, 200_000.0, 300_000.0));
        float[] arrayFloat2_case2 = new float[] { -100_000.0f, -200_000.0f, -300_000.0f };

        innerProd = new KNNScoringSpace.InnerProd(arrayListQueryObject_case2, fieldType);

        assertEquals(7.142857143E-12F, innerProd.scoringMethod.apply(arrayFloat_case2, arrayFloat2_case2), 1.0E-11F);

        float[] arrayFloat_case3 = new float[] { 100_000.0f, 200_000.0f, 300_000.0f };
        List<Double> arrayListQueryObject_case3 = new ArrayList<>(Arrays.asList(100_000.0, 200_000.0, 300_000.0));
        float[] arrayFloat2_case3 = new float[] { 100_000.0f, 200_000.0f, 300_000.0f };

        innerProd = new KNNScoringSpace.InnerProd(arrayListQueryObject_case3, fieldType);

        assertEquals(140_000_000_001F, innerProd.scoringMethod.apply(arrayFloat_case3, arrayFloat2_case3), 0.01F);

        NumberFieldMapper.NumberFieldType invalidFieldType = new NumberFieldMapper.NumberFieldType(
            "field",
            NumberFieldMapper.NumberType.INTEGER
        );
        expectThrows(IllegalArgumentException.class, () -> new KNNScoringSpace.InnerProd(arrayListQueryObject_case2, invalidFieldType));
    }

    @SuppressWarnings("unchecked")
    public void testHammingBit_Long() {
        NumberFieldMapper.NumberFieldType fieldType = new NumberFieldMapper.NumberFieldType("field", NumberFieldMapper.NumberType.LONG);
        Long longObject1 = 1234L; // ..._0000_0100_1101_0010
        Long longObject2 = 2468L; // ..._0000_1001_1010_0100
        KNNScoringSpace.HammingBit hammingBit = new KNNScoringSpace.HammingBit(longObject1, fieldType);

        assertEquals(0.1111F, ((BiFunction<Long, Long, Float>) hammingBit.scoringMethod).apply(longObject1, longObject2), 0.1F);

        KNNVectorFieldMapper.KNNVectorFieldType invalidFieldType = mock(KNNVectorFieldMapper.KNNVectorFieldType.class);
        expectThrows(IllegalArgumentException.class, () -> new KNNScoringSpace.HammingBit(longObject1, invalidFieldType));
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

        KNNVectorFieldMapper.KNNVectorFieldType invalidFieldType = mock(KNNVectorFieldMapper.KNNVectorFieldType.class);
        expectThrows(IllegalArgumentException.class, () -> new KNNScoringSpace.HammingBit(base64Object1, invalidFieldType));
    }
}

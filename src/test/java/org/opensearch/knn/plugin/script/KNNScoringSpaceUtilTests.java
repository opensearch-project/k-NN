/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */
/*
 *   Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License").
 *   You may not use this file except in compliance with the License.
 *   A copy of the License is located at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   or in the "license" file accompanying this file. This file is distributed
 *   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *   express or implied. See the License for the specific language governing
 *   permissions and limitations under the License.
 */

package org.opensearch.knn.plugin.script;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNVectorFieldMapper;
import org.opensearch.index.mapper.BinaryFieldMapper;
import org.opensearch.index.mapper.NumberFieldMapper;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class KNNScoringSpaceUtilTests extends KNNTestCase {
    public void testFieldTypeCheck() {
        assertTrue(KNNScoringSpaceUtil.isLongFieldType(new NumberFieldMapper.NumberFieldType("field",
                NumberFieldMapper.NumberType.LONG)));
        assertFalse(KNNScoringSpaceUtil.isLongFieldType(new NumberFieldMapper.NumberFieldType("field",
                NumberFieldMapper.NumberType.INTEGER)));
        assertFalse(KNNScoringSpaceUtil.isLongFieldType(new BinaryFieldMapper.BinaryFieldType("test")));

        assertTrue(KNNScoringSpaceUtil.isBinaryFieldType(new BinaryFieldMapper.BinaryFieldType("test")));
        assertFalse(KNNScoringSpaceUtil.isBinaryFieldType(new NumberFieldMapper.NumberFieldType("field",
                NumberFieldMapper.NumberType.INTEGER)));

        assertTrue(KNNScoringSpaceUtil.isKNNVectorFieldType(mock(KNNVectorFieldMapper.KNNVectorFieldType.class)));
        assertFalse(KNNScoringSpaceUtil.isKNNVectorFieldType(new BinaryFieldMapper.BinaryFieldType("test")));
    }

    public void testParseLongQuery() {
        int integerQueryObject = 157;
        assertEquals(Long.valueOf(integerQueryObject), KNNScoringSpaceUtil.parseToLong(integerQueryObject));

        Long longQueryObject = 10001L;
        assertEquals(longQueryObject, KNNScoringSpaceUtil.parseToLong(longQueryObject));

        String invalidQueryObject = "invalid";
        expectThrows(IllegalArgumentException.class, () -> KNNScoringSpaceUtil.parseToLong(invalidQueryObject));
    }

    public void testParseBinaryQuery() {
        String base64String = "SrtFZw==";

        /*
         * B64:         "SrtFZw=="
         * Decoded Hex: 4ABB4567
         */

        assertEquals(new BigInteger("4ABB4567", 16), KNNScoringSpaceUtil.parseToBigInteger(base64String));
    }

    public void testParseKNNVectorQuery() {
        float[] arrayFloat = new float[]{1.0f, 2.0f, 3.0f};
        List<Double> arrayListQueryObject = new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0));

        KNNVectorFieldMapper.KNNVectorFieldType fieldType = mock(KNNVectorFieldMapper.KNNVectorFieldType.class);

        when(fieldType.getDimension()).thenReturn(3);
        assertArrayEquals(arrayFloat, KNNScoringSpaceUtil.parseToFloatArray(arrayListQueryObject, 3), 0.1f);

        expectThrows(IllegalStateException.class, () -> KNNScoringSpaceUtil.parseToFloatArray(arrayListQueryObject, 4));

        String invalidObject = "invalidObject";
        expectThrows(ClassCastException.class, () -> KNNScoringSpaceUtil.parseToFloatArray(invalidObject, 3));
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.qframe;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

public class QuantizationConfigParserTests extends KNNTestCase {

    public void testFromCsv() {
        assertEquals(QuantizationConfig.EMPTY, QuantizationConfigParser.fromCsv(""));
        assertEquals(QuantizationConfig.EMPTY, QuantizationConfigParser.fromCsv(null));

        expectThrows(
            IllegalArgumentException.class,
            () -> QuantizationConfigParser.fromCsv(QuantizationConfigParser.TYPE_NAME + "=" + QuantizationConfigParser.BINARY_TYPE)
        );

        expectThrows(
            IllegalArgumentException.class,
            () -> QuantizationConfigParser.fromCsv(
                QuantizationConfigParser.TYPE_NAME + "=invalid," + QuantizationConfigParser.BIT_COUNT_NAME + "=4"
            )
        );

        expectThrows(
            IllegalArgumentException.class,
            () -> QuantizationConfigParser.fromCsv(
                QuantizationConfigParser.TYPE_NAME
                    + "="
                    + QuantizationConfigParser.BINARY_TYPE
                    + ",invalid=4"
                    + QuantizationConfigParser.BIT_COUNT_NAME
                    + "=4"
            )
        );

        expectThrows(
            IllegalArgumentException.class,
            () -> QuantizationConfigParser.fromCsv(
                QuantizationConfigParser.TYPE_NAME
                    + "="
                    + QuantizationConfigParser.BINARY_TYPE
                    + ","
                    + QuantizationConfigParser.BIT_COUNT_NAME
                    + "=invalid"
            )
        );

        expectThrows(
            IllegalArgumentException.class,
            () -> QuantizationConfigParser.fromCsv(
                QuantizationConfigParser.TYPE_NAME
                    + "="
                    + QuantizationConfigParser.BINARY_TYPE
                    + ","
                    + QuantizationConfigParser.BIT_COUNT_NAME
                    + "=4"
                    + ","
                    + QuantizationConfigParser.RANDOM_ROTATION_NAME
                    + "=false"
                    + ","
                    + QuantizationConfigParser.ADC_NAME
                    + "=false"
                    + ","
                    + "OTHER_FIFTH_OPTION"
                    + "=20"
            )
        );

        assertEquals(
            QuantizationConfig.builder().quantizationType(ScalarQuantizationType.FOUR_BIT).build(),
            QuantizationConfigParser.fromCsv(
                QuantizationConfigParser.TYPE_NAME
                    + "="
                    + QuantizationConfigParser.BINARY_TYPE
                    + ","
                    + QuantizationConfigParser.BIT_COUNT_NAME
                    + "=4"
                    + ","
                    + QuantizationConfigParser.RANDOM_ROTATION_NAME
                    + "=false"
                    + ","
                    + QuantizationConfigParser.ADC_NAME
                    + "=false"
            )
        );
    }

    public void testToCsv() {
        assertEquals("", QuantizationConfigParser.toCsv(null));
        assertEquals("", QuantizationConfigParser.toCsv(QuantizationConfig.EMPTY));
        assertEquals(
            "type=binary,bits=2,random_rotation=false,enable_adc=false",
            QuantizationConfigParser.toCsv(QuantizationConfig.builder().quantizationType(ScalarQuantizationType.TWO_BIT).build())
        );
    }

    public void testFromCsv_bwc() {
        assertEquals(
            QuantizationConfig.builder().quantizationType(ScalarQuantizationType.TWO_BIT).enableRandomRotation(false).build(),
            QuantizationConfigParser.fromCsv("type=binary,bits=2")
        );

        assertEquals(
            QuantizationConfig.builder().quantizationType(ScalarQuantizationType.TWO_BIT).enableRandomRotation(false).build(),
            QuantizationConfigParser.fromCsv("type=binary,bits=2")
        );

        assertEquals(
            QuantizationConfig.builder().quantizationType(ScalarQuantizationType.TWO_BIT).enableRandomRotation(false).build(),
            QuantizationConfigParser.fromCsv("type=binary,bits=2,random_rotation=false,enable_adc=false")
        );
    }
}

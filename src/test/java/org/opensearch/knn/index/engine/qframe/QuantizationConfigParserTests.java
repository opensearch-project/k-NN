/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.qframe;

import org.apache.lucene.util.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

public class QuantizationConfigParserTests extends KNNTestCase {

    public void testFromCsv() {
        assertEquals(QuantizationConfig.EMPTY, QuantizationConfigParser.fromCsv("", Version.LATEST));
        assertEquals(QuantizationConfig.EMPTY, QuantizationConfigParser.fromCsv(null, Version.LATEST));

        expectThrows(
            IllegalArgumentException.class,
            () -> QuantizationConfigParser.fromCsv(
                QuantizationConfigParser.TYPE_NAME + "=" + QuantizationConfigParser.BINARY_TYPE,
                Version.LATEST
            )
        );

        expectThrows(
            IllegalArgumentException.class,
            () -> QuantizationConfigParser.fromCsv(
                QuantizationConfigParser.TYPE_NAME + "=invalid," + QuantizationConfigParser.BIT_COUNT_NAME + "=4",
                Version.LATEST
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
                    + "=4",
                Version.LATEST
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
                    + "=invalid",
                Version.LATEST
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
                    + "=false",
                Version.LATEST
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
        // version selected somewhat arbitrarily; 990 is the version w quantization codecs.
        // The important thing is that the version < 10.2.1.
        Version previousVersion990 = Version.LUCENE_9_9_0;

        assertEquals(
            QuantizationConfig.builder().quantizationType(ScalarQuantizationType.TWO_BIT).enableRandomRotation(false).build(),
            QuantizationConfigParser.fromCsv("type=binary,bits=2", previousVersion990)
        );

        // Test lucene > 10.0 but < 10.2.1
        Version previousVersion101 = Version.LUCENE_10_1_0;
        assertEquals(
            QuantizationConfig.builder().quantizationType(ScalarQuantizationType.TWO_BIT).enableRandomRotation(false).build(),
            QuantizationConfigParser.fromCsv("type=binary,bits=2", previousVersion101)
        );

        assertEquals(
            QuantizationConfig.builder().quantizationType(ScalarQuantizationType.TWO_BIT).enableRandomRotation(false).build(),
            QuantizationConfigParser.fromCsv("type=binary,bits=2,random_rotation=false,enable_adc=false", Version.LATEST)
        );
    }
}

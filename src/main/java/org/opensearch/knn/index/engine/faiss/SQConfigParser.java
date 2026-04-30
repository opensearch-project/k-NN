/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.apache.lucene.analysis.util.CSVUtil;

import java.util.Locale;

import static org.opensearch.knn.common.KNNConstants.SQ_BITS;

/**
 * Parser for SQ encoder config stored as a CSV field attribute.
 * Format: {@code bits=1}
 *
 * @see SQConfig
 */
public class SQConfigParser {

    private static final String SEPARATOR = "=";

    /**
     * Serialize an {@link SQConfig} to CSV format for storage as a field attribute.
     */
    public static String toCsv(SQConfig config) {
        if (config == null || config == SQConfig.EMPTY) {
            return "";
        }
        return SQ_BITS + SEPARATOR + config.getBits();
    }

    /**
     * Deserialize a CSV string back to an {@link SQConfig}.
     */
    public static SQConfig fromCsv(String csv) {
        if (csv == null || csv.isEmpty()) {
            return SQConfig.EMPTY;
        }
        String[] csvArray = CSVUtil.parse(csv);
        // Currently only bits is stored. For forward compatibility, accept length >= 1.
        if (csvArray.length < 1) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Invalid csv for SQ config: \"%s\"", csv));
        }
        String bitsValue = getValueOrThrow(SQ_BITS, csvArray[0]);
        return SQConfig.builder().bits(Integer.parseInt(bitsValue)).build();
    }

    private static String getValueOrThrow(String expectedKey, String keyValue) {
        String[] keyValueArr = keyValue.split(SEPARATOR);
        if (keyValueArr.length != 2) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Invalid csv value for SQ config: \"%s\"", keyValue));
        }
        if (keyValueArr[0].equals(expectedKey) == false) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Expected: \"%s\" But got: \"%s\"", expectedKey, keyValue));
        }
        return keyValueArr[1];
    }
}

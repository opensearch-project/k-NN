/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.qframe;

import org.apache.lucene.analysis.util.CSVUtil;
import org.opensearch.knn.index.engine.faiss.QFrameBitEncoder;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

import java.util.List;
import java.util.Locale;

/**
 * Parse util for quantization config
 */
public class QuantizationConfigParser {

    public static final String SEPARATOR = "=";
    public static final String TYPE_NAME = "type";
    public static final String BINARY_TYPE = QFrameBitEncoder.NAME;

    public static final String BYTE_TYPE = "byte";

    public static final List<String> QUANTIZATION_CONFIG_TYPES = List.of(BINARY_TYPE, BYTE_TYPE);
    public static final String BIT_COUNT_NAME = QFrameBitEncoder.BITCOUNT_PARAM;

    /**
     * Parse quantization config to csv format
     * Example: type=binary,bits=2
     * @param quantizationConfig Quantization config
     * @return Csv format of quantization config
     */
    public static String toCsv(QuantizationConfig quantizationConfig) {
        if (quantizationConfig == null
            || quantizationConfig == QuantizationConfig.EMPTY
            || quantizationConfig.getQuantizationType() == null) {
            return "";
        }
        if (quantizationConfig.getQuantizationType() == ScalarQuantizationType.EIGHT_BIT) {
            return TYPE_NAME + SEPARATOR + BYTE_TYPE + "," + BIT_COUNT_NAME + SEPARATOR + quantizationConfig.getQuantizationType().getId();
        }

        return TYPE_NAME + SEPARATOR + BINARY_TYPE + "," + BIT_COUNT_NAME + SEPARATOR + quantizationConfig.getQuantizationType().getId();
    }

    /**
     * Parse csv format to quantization config
     *
     * @param csv Csv format of quantization config
     * @return Quantization config
     */
    public static QuantizationConfig fromCsv(String csv) {
        if (csv == null || csv.isEmpty()) {
            return QuantizationConfig.EMPTY;
        }

        String[] csvArray = CSVUtil.parse(csv);
        if (csvArray.length != 2) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Invalid csv for quantization config: \"%s\"", csv));
        }

        String typeValue = getValueOrThrow(TYPE_NAME, csvArray[0]);
        if (!QUANTIZATION_CONFIG_TYPES.contains(typeValue)) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Unsupported quantization type: \"%s\"", typeValue));
        }

        String bitsValue = getValueOrThrow(BIT_COUNT_NAME, csvArray[1]);
        int bitCount = Integer.parseInt(bitsValue);
        ScalarQuantizationType quantizationType = ScalarQuantizationType.fromId(bitCount);
        return QuantizationConfig.builder().quantizationType(quantizationType).build();
    }

    private static String getValueOrThrow(String expectedKey, String keyValue) {
        String[] keyValueArr = keyValue.split(SEPARATOR);
        if (keyValueArr.length != 2) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Invalid csv value for quantization config: \"%s\"", keyValue));
        }

        if (!keyValueArr[0].equals(expectedKey)) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Expected: \"%s\" But got: \"%s\"", expectedKey, keyValue));
        }

        return keyValueArr[1];
    }
}

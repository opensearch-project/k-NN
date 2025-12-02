/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.qframe;

import org.apache.lucene.analysis.util.CSVUtil;
import org.opensearch.Version;
import org.opensearch.knn.index.engine.faiss.QFrameBitEncoder;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

import java.util.Locale;

/**
 * Parse util for quantization config
 */
public class QuantizationConfigParser {

    public static final String SEPARATOR = "=";
    public static final String TYPE_NAME = "type";
    public static final String BINARY_TYPE = QFrameBitEncoder.NAME;
    public static final String BIT_COUNT_NAME = QFrameBitEncoder.BITCOUNT_PARAM;
    public static final String RANDOM_ROTATION_NAME = QFrameBitEncoder.ENABLE_RANDOM_ROTATION_PARAM;
    public static final String ADC_NAME = QFrameBitEncoder.ENABLE_ADC_PARAM;

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
        String result = TYPE_NAME
            + SEPARATOR
            + BINARY_TYPE
            + ","
            + BIT_COUNT_NAME
            + SEPARATOR
            + quantizationConfig.getQuantizationType().getId();

        if (Version.CURRENT.onOrAfter(Version.V_3_2_0)) {
            result = result
                + ","
                + RANDOM_ROTATION_NAME
                + SEPARATOR
                + quantizationConfig.isEnableRandomRotation()
                + ","
                + ADC_NAME
                + SEPARATOR
                + quantizationConfig.isEnableADC();
            ;
        }

        return result;
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
        int csvArrayLength = csvArray.length;

        // if length is not 2 or not 4 then the csv is invalid.
        if (csvArrayLength < 2) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Invalid csv (length < 2) for quantization config: \"%s\"", csv));
        }

        // Parse common fields (type and bits)
        String typeValue = getValueOrThrow(TYPE_NAME, csvArray[0]);
        if (!typeValue.equals(BINARY_TYPE)) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Unsupported quantization type: \"%s\"", typeValue));
        }

        String bitsValue = getValueOrThrow(BIT_COUNT_NAME, csvArray[1]);
        int bitCount = Integer.parseInt(bitsValue);
        ScalarQuantizationType quantizationType = ScalarQuantizationType.fromId(bitCount);

        // RR is disabled by default, and it must be disabled for old segments since the extra quantization info is not present.
        boolean isEnableRandomRotation = QFrameBitEncoder.DEFAULT_ENABLE_RANDOM_ROTATION;
        // ADC is disabled by default, and it must be disabled for old segments since the extra quantization info is not present.
        boolean isEnableADC = QFrameBitEncoder.DEFAULT_ENABLE_ADC;

        // parse "random_rotation" and "enable_adc" from csv if length 4
        if (csvArrayLength == 4) {
            String isEnableRandomRotationValue = getValueOrThrow(RANDOM_ROTATION_NAME, csvArray[2]);
            isEnableRandomRotation = Boolean.parseBoolean(isEnableRandomRotationValue);

            String isEnableADCValue = getValueOrThrow(ADC_NAME, csvArray[3]);
            isEnableADC = Boolean.parseBoolean(isEnableADCValue);
        } else if (csvArrayLength != 2) {
            // length == 3 or length > 4. Both cases are invalid.
            // For forward compatability we will reserve length > 4 lists for new config options.
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "Invalid csv (length must be 2 or 4) for quantization config: \"%s\"", csv)
            );
        }

        return QuantizationConfig.builder()
            .quantizationType(quantizationType)
            .enableRandomRotation(isEnableRandomRotation)
            .enableADC(isEnableADC)
            .build();
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

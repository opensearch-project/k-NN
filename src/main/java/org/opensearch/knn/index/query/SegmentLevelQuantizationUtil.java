/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.experimental.UtilityClass;
import org.apache.lucene.index.LeafReader;
import org.opensearch.knn.index.codec.KNN990Codec.QuantizationConfigKNNCollector;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;
import java.util.Locale;

/**
 * A utility class for doing Quantization related operation at a segment level. We can move this utility in {@link SegmentLevelQuantizationInfo}
 * but I am keeping it thinking that {@link SegmentLevelQuantizationInfo} free from these utility functions to reduce
 * the responsibilities of {@link SegmentLevelQuantizationInfo} class.
 */
@UtilityClass
public class SegmentLevelQuantizationUtil {

    public static boolean isAdcEnabled(SegmentLevelQuantizationInfo segmentLevelQuantizationInfo) {
        return segmentLevelQuantizationInfo != null
            && ScalarQuantizationParams.generateTypeIdentifier(ScalarQuantizationType.ONE_BIT)
                .equals(segmentLevelQuantizationInfo.getQuantizationParams().getTypeIdentifier());
    }

    /**
     * A simple function to convert a vector to a quantized vector for a segment.
     * @param vector array of float
     * @return array of byte
     */
    @SuppressWarnings("unchecked")
    public static byte[] quantizeVector(final float[] vector, final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo) {
        if (segmentLevelQuantizationInfo == null) {
            return null;
        }
        final QuantizationService quantizationService = QuantizationService.getInstance();
        // TODO: We are converting the output of Quantize to byte array for now. But this needs to be fixed when
        // other types of quantized outputs are returned like float[].
        return (byte[]) quantizationService.quantize(
            segmentLevelQuantizationInfo.getQuantizationState(),
            vector,
            quantizationService.createQuantizationOutput(segmentLevelQuantizationInfo.getQuantizationParams())
        );
    }

    public static void transformVector(final float[] vector, final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo) {
        if (segmentLevelQuantizationInfo == null) {
            return;
        }
        final QuantizationService quantizationService = QuantizationService.getInstance();
        quantizationService.transform(segmentLevelQuantizationInfo.getQuantizationState(), vector);
    }

    /**
     * A utility function to get {@link QuantizationState} for a given segment and field.
     * @param leafReader {@link LeafReader}
     * @param fieldName {@link String}
     * @return {@link QuantizationState}
     * @throws IOException exception during reading the {@link QuantizationState}
     */
    static QuantizationState getQuantizationState(final LeafReader leafReader, String fieldName) throws IOException {
        final QuantizationConfigKNNCollector tempCollector = new QuantizationConfigKNNCollector();
        leafReader.searchNearestVectors(fieldName, new float[0], tempCollector, null);
        if (tempCollector.getQuantizationState() == null) {
            throw new IllegalStateException(String.format(Locale.ROOT, "No quantization state found for field %s", fieldName));
        }
        return tempCollector.getQuantizationState();
    }
}

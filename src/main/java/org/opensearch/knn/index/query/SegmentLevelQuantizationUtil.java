/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.experimental.UtilityClass;
import org.apache.lucene.index.LeafReader;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.KNN990Codec.QuantizationConfigKNNCollector;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
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
        if (segmentLevelQuantizationInfo == null) return false;
        return isAdcEnabled(segmentLevelQuantizationInfo.getQuantizationParams());
    }

    public static boolean isAdcEnabled(QuantizationParams segmentLevelQuantizationParams) {
        if (segmentLevelQuantizationParams instanceof ScalarQuantizationParams scalarQuantizationParams) {
            return scalarQuantizationParams.isEnableADC();
        } else {
            return false;
        }
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

    /**
     * Transform vector with ADC. ADC allows us to score full-precision query vectors against binary document vectors.
     * The transformation formula is:
     * q_d = (q_d - x_d) / (y_d - x_d) where x_d is the mean of all document entries quantized to 0 (the below threshold mean)
     * and y_d is the mean of all document entries quantized to 1 (the above threshold mean).
     * @param vector array of floats, modified in-place.
     * @param segmentLevelQuantizationInfo quantization state including below and above threshold means to perform the transformation.
     * @param spaceType spaceType (l2 or innerproduct). Used to identify whether an additional correction term should be applied.
     */
    public static void transformVectorWithADC(
        float[] vector,
        final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo,
        SpaceType spaceType
    ) {
        if (segmentLevelQuantizationInfo == null) {
            return;
        }
        final QuantizationService quantizationService = QuantizationService.getInstance();
        quantizationService.transformWithADC(segmentLevelQuantizationInfo.getQuantizationState(), vector, spaceType);
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

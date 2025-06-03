/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.util.Version;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;

/**
 * This class encapsulate the necessary details to do the quantization of the vectors present in a lucene segment.
 */
@Getter
@RequiredArgsConstructor(access = AccessLevel.PRIVATE)
public class SegmentLevelQuantizationInfo {
    private final QuantizationParams quantizationParams;
    private final QuantizationState quantizationState;

    /**
     * A builder like function to build the {@link SegmentLevelQuantizationInfo}
     * @param leafReader {@link LeafReader}
     * @param fieldInfo {@link FieldInfo}
     * @param fieldName {@link String}
     * @return {@link SegmentLevelQuantizationInfo}
     * @throws IOException exception while creating the {@link SegmentLevelQuantizationInfo} object.
     */
    public static SegmentLevelQuantizationInfo build(
        final LeafReader leafReader,
        final FieldInfo fieldInfo,
        final String fieldName,
        Version luceneVersion
    ) throws IOException {
        // TODO: here3

        final QuantizationParams quantizationParams = QuantizationService.getInstance().getQuantizationParams(fieldInfo, luceneVersion);
        if (quantizationParams == null) {
            return null;
        }
        final QuantizationState quantizationState = SegmentLevelQuantizationUtil.getQuantizationState(leafReader, fieldName);

        return new SegmentLevelQuantizationInfo(quantizationParams, quantizationState);
    }

}

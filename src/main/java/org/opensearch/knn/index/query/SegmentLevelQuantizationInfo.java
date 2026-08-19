/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReader;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;

/**
 * This class encapsulate the necessary details to do the quantization of the vectors present in a lucene segment.
 */
@Getter
@RequiredArgsConstructor(access = AccessLevel.PRIVATE)
@Log4j2
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
    public static SegmentLevelQuantizationInfo build(final LeafReader leafReader, final FieldInfo fieldInfo, final String fieldName)
        throws IOException {
        // Since we don't know top code has made sure that field is vector or not, we are doing this check to ensure that
        // we validate it.
        if (fieldInfo == null || fieldInfo.hasVectorValues() == false) {
            log.debug("The segment field {} is not a vector field.", fieldName);
            return null;
        }

        final QuantizationParams quantizationParams = QuantizationService.getInstance().getQuantizationParams(fieldInfo);
        if (quantizationParams == null) {
            return null;
        }

        // Before getting the quantization state check if this segment has vector docs or not. Just by checking fieldInfo
        // you cannot make sure that it has docs. So we are validating it with FloatVectorValues.
        final FloatVectorValues fvv = leafReader.getFloatVectorValues(fieldName);
        if (fvv == null || fvv.size() == 0) {
            log.debug("The segment has BQ field with name : {}, but no vectors in it.", fieldName);
            return null;
        }

        final QuantizationState quantizationState = SegmentLevelQuantizationUtil.getQuantizationState(leafReader, fieldName);
        return new SegmentLevelQuantizationInfo(quantizationParams, quantizationState);
    }
}

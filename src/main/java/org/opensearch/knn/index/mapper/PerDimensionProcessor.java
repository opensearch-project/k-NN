/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.clipVectorValueToFP16Range;

/**
 * Process values per dimension. Good to have if we want to do some kind of cleanup on data as it is coming in.
 */
public interface PerDimensionProcessor {

    /**
     * Process float value per dimension.
     *
     * @param value value to process
     * @return processed value
     */
    default float process(float value) {
        return value;
    }

    /**
     * Process byte as float value per dimension.
     *
     * @param value value to process
     * @return processed value
     */
    default float processByte(float value) {
        return value;
    }

    PerDimensionProcessor NOOP_PROCESSOR = new PerDimensionProcessor() {
    };

    // If the encoder parameter, "clip" is set to True, if the vector value is outside the FP16 range then it will be
    // clipped to FP16 range.
    PerDimensionProcessor CLIP_TO_FP16_PROCESSOR = new PerDimensionProcessor() {
        @Override
        public float process(float value) {
            return clipVectorValueToFP16Range(value);
        }

        @Override
        public float processByte(float value) {
            throw new IllegalStateException("CLIP_TO_FP16_PROCESSOR should not be called with byte type");
        }
    };
}

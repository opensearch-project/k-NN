/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.commons.lang3.StringUtils;
import org.apache.lucene.index.SegmentInfo;

import java.util.Arrays;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Helper class for adding information into the segment attributes
 */
public class DerivedSourceSegmentAttributeParser {

    static final String DERIVED_SOURCE_FIELD = "derived_vector_fields";
    static final String NESTED_DERIVED_SOURCE_FIELD = "nested_derived_vector_fields";
    static final String DELIMETER = ",";

    /**
     * From segmentInfo, parse the derived_vector_fields
     *
     * @param segmentInfo {@link SegmentInfo}
     * @param isNested Whether the vector field is nested or not
     * @return List of fields that derived source is enabled for. Potentially null if no fields
     */
    public static List<String> parseDerivedVectorFields(SegmentInfo segmentInfo, boolean isNested) {
        if (segmentInfo == null) {
            return Collections.emptyList();
        }
        String fieldName = isNested ? NESTED_DERIVED_SOURCE_FIELD : DERIVED_SOURCE_FIELD;
        String derivedVectorFields = segmentInfo.getAttribute(fieldName);
        if (StringUtils.isEmpty(derivedVectorFields)) {
            return Collections.emptyList();
        }
        return Arrays.stream(derivedVectorFields.split(DELIMETER, -1)).collect(Collectors.toList());
    }

    /**
     * Adds {@link SegmentInfo} attribute for vectorFieldTypes
     *
     * @param segmentInfo {@link SegmentInfo}
     * @param vectorFieldTypes List of vector field names
     * @param isNested Whether the vector field is nested or not
     */
    public static void addDerivedVectorFieldsSegmentInfoAttribute(
        SegmentInfo segmentInfo,
        List<String> vectorFieldTypes,
        boolean isNested
    ) {
        if (segmentInfo == null) {
            throw new IllegalArgumentException("SegmentInfo cannot be null");
        }
        if (vectorFieldTypes == null || vectorFieldTypes.isEmpty()) {
            throw new IllegalArgumentException("VectorFieldTypes cannot be null or empty");
        }
        String fieldName = isNested ? NESTED_DERIVED_SOURCE_FIELD : DERIVED_SOURCE_FIELD;
        segmentInfo.putAttribute(fieldName, String.join(DELIMETER, vectorFieldTypes));
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.index.SegmentInfo;

import java.util.Arrays;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Helper class for adding information into the segment attributes
 */
public class DerivedSourceSegmentAttributeHelper {

    public static final String DERIVED_SOURCE_FIELD = "derived_vector_fields";

    /**
     * From segmentInfo, parse the derived_vector_fields
     *
     * @param segmentInfo {@link SegmentInfo}
     * @return List of fields that derived source is enabled for. Potentially null if no fields
     */
    public static List<String> parseDerivedVectorFields(SegmentInfo segmentInfo) {
        if (segmentInfo == null) {
            return Collections.emptyList();
        }
        String derivedVectorFields = segmentInfo.getAttribute(DERIVED_SOURCE_FIELD);
        if (derivedVectorFields == null || derivedVectorFields.isEmpty()) {
            return Collections.emptyList();
        }
        return Arrays.stream(derivedVectorFields.split(",")).collect(Collectors.toList());
    }

    /**
     * Adds {@link SegmentInfo} attribute for vectorFieldTypes
     *
     * @param segmentInfo {@link SegmentInfo}
     * @param vectorFieldTypes List of vector field names
     */
    public static void addDerivedVectorFieldsSegmentInfoAttribute(SegmentInfo segmentInfo, List<String> vectorFieldTypes) {
        segmentInfo.putAttribute(DERIVED_SOURCE_FIELD, String.join(",", vectorFieldTypes));
    }
}

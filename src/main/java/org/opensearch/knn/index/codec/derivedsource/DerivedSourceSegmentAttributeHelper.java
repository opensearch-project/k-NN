/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Helper class for adding information into the segment attributes
 */
public class DerivedSourceSegmentAttributeHelper {

    public static final String DERIVED_SOURCE_FIELD = "derived_vector_fields";
    public static final String NESTED_LINEAGE = "derived_vector_fields_nested_lineage";

    /**
     * From segmentInfo, parse the derived_vector_fields
     *
     * @param segmentInfo {@link SegmentInfo}
     * @param fieldInfos {@link FieldInfo}
     * @return List of fields that derived source is enabled for. Potentially null if no fields
     */
    public static List<FieldInfo> parseDerivedVectorFields(SegmentInfo segmentInfo, FieldInfos fieldInfos) {
        if (segmentInfo == null) {
            return null;
        }
        String derivedVectorFields = segmentInfo.getAttribute(DERIVED_SOURCE_FIELD);
        if (derivedVectorFields == null || derivedVectorFields.isEmpty()) {
            return null;
        }
        return Arrays.stream(derivedVectorFields.split(","))
            .filter(field -> fieldInfos.fieldInfo(field) != null)
            .map(fieldInfos::fieldInfo)
            .collect(Collectors.toList());
    }

    /**
     * Parses the nested lineage map from the segment info
     *
     * @param vectorFields Fields of vectors
     * @param segmentInfo {@link SegmentInfo}
     * @return Mapping between derived source fields and their nested lineage
     */
    public static Map<String, List<String>> parseNestedLineageMap(List<FieldInfo> vectorFields, SegmentInfo segmentInfo) {
        if (vectorFields == null || vectorFields.isEmpty()) {
            return Collections.emptyMap();
        }
        String derivedVectorFieldsLineage = segmentInfo.getAttribute(NESTED_LINEAGE);
        if (derivedVectorFieldsLineage == null || derivedVectorFieldsLineage.isEmpty()) {
            return Collections.emptyMap();
        }
        List<String> nestedLineageStrings = Arrays.asList(derivedVectorFieldsLineage.split(",", -1));
        Map<String, List<String>> nestedLineageMap = new HashMap<>();
        for (int i = 0; i < vectorFields.size(); i++) {
            nestedLineageMap.put(
                vectorFields.get(i).name,
                Arrays.stream(nestedLineageStrings.get(i).split(";")).filter(s -> s.isEmpty() == false).toList()
            );
        }
        return nestedLineageMap;
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

    /**
     * Adds {@link SegmentInfo} attribute for nested lineage of all derived source fields
     *
     * @param segmentInfo {@link SegmentInfo}
     * @param nestedLineageForAllFields List of lists of parent and grandparent fields. Order should match that of the
     *                                  vector field types
     */
    public static void addNestedLineageSegmentInfoAttribute(SegmentInfo segmentInfo, List<List<String>> nestedLineageForAllFields) {
        segmentInfo.putAttribute(
            NESTED_LINEAGE,
            nestedLineageForAllFields.stream().map(f -> String.join(";", f)).collect(Collectors.joining(","))
        );
    }
}

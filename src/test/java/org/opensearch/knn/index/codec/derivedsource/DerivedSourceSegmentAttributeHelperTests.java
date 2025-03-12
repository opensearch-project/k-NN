/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.mockito.Mock;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.codec.derivedsource.DerivedSourceSegmentAttributeHelper.DERIVED_SOURCE_FIELD;
import static org.opensearch.knn.index.codec.derivedsource.DerivedSourceSegmentAttributeHelper.NESTED_LINEAGE;

public class DerivedSourceSegmentAttributeHelperTests extends KNNTestCase {

    @Mock
    SegmentInfo mockSegmentInfo;

    public void testAddDerivedVectorFieldsSegmentInfoAttribue() {
        Map<String, String> fakeAttributes = new HashMap<>();
        when(mockSegmentInfo.putAttribute(any(), any())).thenAnswer(t -> fakeAttributes.put(t.getArgument(0), t.getArgument(1)));

        DerivedSourceSegmentAttributeHelper.addDerivedVectorFieldsSegmentInfoAttribute(
            mockSegmentInfo,
            List.of("test", "test2.nested", "vector")
        );
        assertEquals("test,test2.nested,vector", fakeAttributes.get(DERIVED_SOURCE_FIELD));
        fakeAttributes.remove(DERIVED_SOURCE_FIELD);

        DerivedSourceSegmentAttributeHelper.addDerivedVectorFieldsSegmentInfoAttribute(mockSegmentInfo, List.of("test"));
        assertEquals("test", fakeAttributes.get(DERIVED_SOURCE_FIELD));
    }

    public void testParseDerivedVectorFields() {
        Map<String, String> fakeAttributes = new HashMap<>();
        when(mockSegmentInfo.getAttribute(any())).thenAnswer(t -> fakeAttributes.get(t.getArgument(0)));

        assertNull(DerivedSourceSegmentAttributeHelper.parseDerivedVectorFields(null, null));
        assertNull(DerivedSourceSegmentAttributeHelper.parseDerivedVectorFields(mockSegmentInfo, null));

        fakeAttributes.put(DERIVED_SOURCE_FIELD, "test,test2.nested,vector");
        List<FieldInfo> expectedFieldInfos = Arrays.asList(
            KNNCodecTestUtil.FieldInfoBuilder.builder("test").fieldNumber(0).build(),
            KNNCodecTestUtil.FieldInfoBuilder.builder("test2.nested").fieldNumber(1).build(),
            KNNCodecTestUtil.FieldInfoBuilder.builder("vector").fieldNumber(2).build()
        );

        List<FieldInfo> fieldInfos = new ArrayList<>(
            Arrays.asList(
                KNNCodecTestUtil.FieldInfoBuilder.builder("test4").fieldNumber(3).build(),
                KNNCodecTestUtil.FieldInfoBuilder.builder("test8.nested").fieldNumber(4).build(),
                KNNCodecTestUtil.FieldInfoBuilder.builder("vector7").fieldNumber(5).build()
            )
        );
        fieldInfos.addAll(expectedFieldInfos);
        assertEquals(
            expectedFieldInfos,
            DerivedSourceSegmentAttributeHelper.parseDerivedVectorFields(
                mockSegmentInfo,
                new FieldInfos(fieldInfos.toArray(new FieldInfo[0]))
            )
        );
    }

    public void testAddNestedLineageSegmentInfoAttribute() {
        Map<String, String> fakeAttributes = new HashMap<>();
        when(mockSegmentInfo.putAttribute(any(), any())).thenAnswer(t -> fakeAttributes.put(t.getArgument(0), t.getArgument(1)));

        DerivedSourceSegmentAttributeHelper.addNestedLineageSegmentInfoAttribute(
            mockSegmentInfo,
            List.of(
                List.of("test1.nested_1.nested_2.nested_3", "test1.nested_1", "test1"),
                List.of("test2.nested_1.nested_2.nested_3", "test2"),
                Collections.emptyList(),
                List.of("test3")
            )
        );
        assertEquals(
            "test1.nested_1.nested_2.nested_3;test1.nested_1;test1,test2.nested_1.nested_2.nested_3;test2,,test3",
            fakeAttributes.get(NESTED_LINEAGE)
        );
    }

    public void testParseNestedLineageMap() {
        Map<String, String> fakeAttributes = new HashMap<>();
        fakeAttributes.put(
            NESTED_LINEAGE,
            "test1.nested_1.nested_2.nested_3;test1.nested_1;test1,test2.nested_1.nested_2.nested_3;test3,,test5"
        );
        when(mockSegmentInfo.getAttribute(any())).thenAnswer(t -> fakeAttributes.get(t.getArgument(0)));
        List<FieldInfo> fieldInfos = Arrays.asList(
            KNNCodecTestUtil.FieldInfoBuilder.builder("test1.nested_1.nested_2.nested_3.test").fieldNumber(0).build(),
            KNNCodecTestUtil.FieldInfoBuilder.builder("test2.nested_1.nested_2.nested_3.test2.nested").fieldNumber(1).build(),
            KNNCodecTestUtil.FieldInfoBuilder.builder("test3.vector").fieldNumber(2).build(),
            KNNCodecTestUtil.FieldInfoBuilder.builder("test4").fieldNumber(3).build(),
            KNNCodecTestUtil.FieldInfoBuilder.builder("test5.nested").fieldNumber(4).build()
        );

        Map<String, List<String>> expectedNestedLineageMap = Map.of(
            "test1.nested_1.nested_2.nested_3.test",
            List.of("test1.nested_1.nested_2.nested_3", "test1.nested_1", "test1"),
            "test2.nested_1.nested_2.nested_3.test2.nested",
            List.of("test1.nested_1", "test1"),
            "test3.vector",
            List.of("test3"),
            "test4",
            Collections.emptyList(),
            "test5.nested",
            List.of("test5")
        );

        assertEquals(expectedNestedLineageMap, DerivedSourceSegmentAttributeHelper.parseNestedLineageMap(fieldInfos, mockSegmentInfo));
    }
}

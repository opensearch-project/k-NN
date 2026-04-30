/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.index.SegmentInfo;
import org.mockito.Mock;
import org.opensearch.knn.KNNTestCase;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.codec.derivedsource.DerivedSourceSegmentAttributeParser.DERIVED_SOURCE_FIELD;

public class DerivedSourceSegmentAttributeParserTests extends KNNTestCase {

    @Mock
    SegmentInfo mockSegmentInfo;

    public void testAddDerivedVectorFieldsSegmentInfoAttribute() {
        Map<String, String> fakeAttributes = new HashMap<>();
        when(mockSegmentInfo.putAttribute(any(), any())).thenAnswer(t -> fakeAttributes.put(t.getArgument(0), t.getArgument(1)));

        DerivedSourceSegmentAttributeParser.addDerivedVectorFieldsSegmentInfoAttribute(
            mockSegmentInfo,
            Set.of("test", "test2.nested", "vector"),
            false
        );
        Set<String> expected = Set.of("test", "test2.nested", "vector");
        Set<String> actual = Set.of(fakeAttributes.get(DERIVED_SOURCE_FIELD).split(","));
        assertEquals(expected, actual);
        fakeAttributes.remove(DERIVED_SOURCE_FIELD);

        DerivedSourceSegmentAttributeParser.addDerivedVectorFieldsSegmentInfoAttribute(mockSegmentInfo, Set.of("test"), false);
        assertEquals("test", fakeAttributes.get(DERIVED_SOURCE_FIELD));

        DerivedSourceSegmentAttributeParser.addDerivedVectorFieldsSegmentInfoAttribute(
            mockSegmentInfo,
            Set.of("a", "", "c", "d", "e"),
            false
        );
        Set<String> expected2 = Set.of("a", "", "c", "d", "e");
        Set<String> actual2 = Set.of(fakeAttributes.get(DERIVED_SOURCE_FIELD).split(",", -1));
        assertEquals(expected2, actual2);
    }

    public void testParseDerivedVectorFields() {
        Map<String, String> fakeAttributes = new HashMap<>();
        when(mockSegmentInfo.getAttribute(any())).thenAnswer(t -> fakeAttributes.get(t.getArgument(0)));

        assertEquals(Collections.emptyList(), DerivedSourceSegmentAttributeParser.parseDerivedVectorFields(null, false));
        assertEquals(Collections.emptyList(), DerivedSourceSegmentAttributeParser.parseDerivedVectorFields(mockSegmentInfo, false));

        fakeAttributes.put(DERIVED_SOURCE_FIELD, "test,test2.nested,vector");
        List<String> expectedFieldInfos = Arrays.asList("test", "test2.nested", "vector");
        assertEquals(expectedFieldInfos, DerivedSourceSegmentAttributeParser.parseDerivedVectorFields(mockSegmentInfo, false));

        fakeAttributes.put(DERIVED_SOURCE_FIELD, ",,,,");
        expectedFieldInfos = Arrays.asList("", "", "", "", "");
        assertEquals(expectedFieldInfos, DerivedSourceSegmentAttributeParser.parseDerivedVectorFields(mockSegmentInfo, false));
    }
}

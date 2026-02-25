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
            List.of("test", "test2.nested", "vector"),
            false
        );
        assertEquals("test,test2.nested,vector", fakeAttributes.get(DERIVED_SOURCE_FIELD));
        fakeAttributes.remove(DERIVED_SOURCE_FIELD);

        DerivedSourceSegmentAttributeParser.addDerivedVectorFieldsSegmentInfoAttribute(mockSegmentInfo, List.of("test"), false);
        assertEquals("test", fakeAttributes.get(DERIVED_SOURCE_FIELD));

        DerivedSourceSegmentAttributeParser.addDerivedVectorFieldsSegmentInfoAttribute(mockSegmentInfo, List.of("", "", "", "", ""), false);
        assertEquals(",,,,", fakeAttributes.get(DERIVED_SOURCE_FIELD));
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

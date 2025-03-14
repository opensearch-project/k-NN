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
import static org.opensearch.knn.index.codec.derivedsource.DerivedSourceSegmentAttributeHelper.DERIVED_SOURCE_FIELD;

public class DerivedSourceSegmentAttributeHelperTests extends KNNTestCase {

    @Mock
    SegmentInfo mockSegmentInfo;

    public void testAddDerivedVectorFieldsSegmentInfoAttribute() {
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

        assertEquals(Collections.emptyList(), DerivedSourceSegmentAttributeHelper.parseDerivedVectorFields(null));
        assertEquals(Collections.emptyList(), DerivedSourceSegmentAttributeHelper.parseDerivedVectorFields(mockSegmentInfo));

        fakeAttributes.put(DERIVED_SOURCE_FIELD, "test,test2.nested,vector");
        List<String> expectedFieldInfos = Arrays.asList("test", "test2.nested", "vector");
        assertEquals(expectedFieldInfos, DerivedSourceSegmentAttributeHelper.parseDerivedVectorFields(mockSegmentInfo));
    }
}

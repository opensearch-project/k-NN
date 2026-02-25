/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentReadState;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class DerivedSourceVectorTransformerTests extends OpenSearchTestCase {

    private DerivedSourceReaders mockDerivedSourceReaders;
    private SegmentReadState mockSegmentReadState;

    private AutoCloseable mocks;

    private static final String[] ALL_FIELDS = { "test_vector", "temp_vector", "user_vector" };

    @Override
    public void setUp() throws Exception {
        super.setUp();
        mocks = MockitoAnnotations.openMocks(this);
        mockDerivedSourceReaders = Mockito.mock(DerivedSourceReaders.class);
        mockSegmentReadState = Mockito.mock(SegmentReadState.class);
    }

    @Override
    public void tearDown() throws Exception {
        super.tearDown();
        mocks.close();
    }

    public void testInitialize_withVariousIncludeExcludeCombinations() {
        // Test 1: No filtering - all fields remain
        assertFieldFiltering(null, null, new String[] { "test_vector", "temp_vector", "user_vector" }, new String[] {});

        // Test 2: Empty includes - all fields remain
        assertFieldFiltering(new String[] {}, null, new String[] { "test_vector", "temp_vector", "user_vector" }, new String[] {});

        // Test 3: Empty excludes - all fields remain
        assertFieldFiltering(null, new String[] {}, new String[] { "test_vector", "temp_vector", "user_vector" }, new String[] {});

        // Test 4: Both empty - all fields remain
        assertFieldFiltering(
            new String[] {},
            new String[] {},
            new String[] { "test_vector", "temp_vector", "user_vector" },
            new String[] {}
        );

        // Test 5: Only includes - only matching fields remain
        assertFieldFiltering(
            new String[] { "test_vector", "user_vector" },
            null,
            new String[] { "test_vector", "user_vector" },
            new String[] { "temp_vector" }
        );

        // Test 6: Only excludes - matching fields removed
        assertFieldFiltering(
            null,
            new String[] { "test_vector" },
            new String[] { "temp_vector", "user_vector" },
            new String[] { "test_vector" }
        );

        // Test 7: Both includes and excludes - excludes override includes
        assertFieldFiltering(
            new String[] { "test_vector", "temp_vector" },
            new String[] { "temp_vector" },
            new String[] { "test_vector" },
            new String[] { "temp_vector", "user_vector" }
        );

        // Test 8: Wildcard includes - only matching fields remain
        assertFieldFiltering(new String[] { "t*" }, null, new String[] { "test_vector", "temp_vector" }, new String[] { "user_vector" });

        // Test 9: Wildcard excludes - matching fields removed
        assertFieldFiltering(null, new String[] { "t*" }, new String[] { "user_vector" }, new String[] { "test_vector", "temp_vector" });

        // Test 10: Wildcard includes with specific excludes
        assertFieldFiltering(
            new String[] { "t*", "user_vector" },
            new String[] { "test_vector" },
            new String[] { "temp_vector", "user_vector" },
            new String[] { "test_vector" }
        );

        // Test 11: All fields excluded with wildcard
        assertFieldFiltering(null, new String[] { "*" }, new String[] {}, new String[] { "test_vector", "temp_vector", "user_vector" });

        // Test 12: Includes match nothing - no fields remain
        assertFieldFiltering(
            new String[] { "nonexistent_*" },
            null,
            new String[] {},
            new String[] { "test_vector", "temp_vector", "user_vector" }
        );

        // Test 13: Excludes match nothing - all fields remain
        assertFieldFiltering(
            null,
            new String[] { "nonexistent_*" },
            new String[] { "test_vector", "temp_vector", "user_vector" },
            new String[] {}
        );
    }

    private void assertFieldFiltering(String[] includes, String[] excludes, String[] expectedPresent, String[] expectedAbsent) {
        DerivedSourceVectorTransformer transformer = createTransformerWithFields(ALL_FIELDS);
        transformer.initialize(includes, excludes);

        Set<String> remainingFields = getRemainingFields(transformer);

        for (String field : expectedPresent) {
            assertTrue(
                String.format(
                    "Field '%s' should be present (includes=%s, excludes=%s)",
                    field,
                    Arrays.toString(includes),
                    Arrays.toString(excludes)
                ),
                remainingFields.contains(field)
            );
        }

        for (String field : expectedAbsent) {
            assertFalse(
                String.format(
                    "Field '%s' should be absent (includes=%s, excludes=%s)",
                    field,
                    Arrays.toString(includes),
                    Arrays.toString(excludes)
                ),
                remainingFields.contains(field)
            );
        }

        assertEquals(
            String.format("Field count mismatch (includes=%s, excludes=%s)", Arrays.toString(includes), Arrays.toString(excludes)),
            expectedPresent.length,
            remainingFields.size()
        );
    }

    private DerivedSourceVectorTransformer createTransformerWithFields(String... fieldNames) {
        try (
            MockedStatic<PerFieldDerivedVectorTransformerFactory> factoryMock = Mockito.mockStatic(
                PerFieldDerivedVectorTransformerFactory.class
            )
        ) {

            factoryMock.when(
                () -> PerFieldDerivedVectorTransformerFactory.create(
                    Mockito.any(FieldInfo.class),
                    Mockito.anyBoolean(),
                    Mockito.any(DerivedSourceReaders.class)
                )
            ).thenReturn(Mockito.mock(PerFieldDerivedVectorTransformer.class));

            List<DerivedFieldInfo> fieldInfos = Arrays.stream(fieldNames).map(this::createMockDerivedFieldInfo).toList();

            return new DerivedSourceVectorTransformer(mockDerivedSourceReaders, mockSegmentReadState, fieldInfos);
        }
    }

    private DerivedFieldInfo createMockDerivedFieldInfo(String name) {
        DerivedFieldInfo mockFieldInfo = Mockito.mock(DerivedFieldInfo.class);
        Mockito.when(mockFieldInfo.name()).thenReturn(name);
        Mockito.when(mockFieldInfo.isNested()).thenReturn(false);
        Mockito.when(mockFieldInfo.fieldInfo()).thenReturn(Mockito.mock(FieldInfo.class));
        return mockFieldInfo;
    }

    private Set<String> getRemainingFields(DerivedSourceVectorTransformer transformer) {
        try {
            java.lang.reflect.Field field = DerivedSourceVectorTransformer.class.getDeclaredField("perFieldDerivedVectorTransformers");
            field.setAccessible(true);
            @SuppressWarnings("unchecked")
            Map<String, ?> map = (Map<String, ?>) field.get(transformer);
            return map.keySet();
        } catch (Exception e) {
            throw new RuntimeException("Failed to access perFieldDerivedVectorTransformers", e);
        }
    }
}

/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.util;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNMethod;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.MethodComponent;
import org.opensearch.knn.index.SpaceType;
import com.google.common.collect.ImmutableMap;
import org.opensearch.common.ValidationException;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import java.util.function.Function;

import static org.opensearch.knn.common.KNNConstants.NAME;

public class KNNLibraryTests extends KNNTestCase {
    /**
     * Test native library build version getter
     */
    public void testNativeLibrary_getLatestBuildVersion() {
        String latestBuildVersion = "test-build-version";
        TestNativeLibrary testNativeLibrary = new TestNativeLibrary(Collections.emptyMap(), Collections.emptyMap(),
                latestBuildVersion, "", "");
        assertEquals(latestBuildVersion, testNativeLibrary.getLatestBuildVersion());
    }

    /**
     * Test native library version getter
     */
    public void testNativeLibrary_getLatestLibVersion() {
        String latestVersion = "test-lib-version";
        TestNativeLibrary testNativeLibrary = new TestNativeLibrary(Collections.emptyMap(), Collections.emptyMap(),
                "", latestVersion, "");
        assertEquals(latestVersion, testNativeLibrary.getLatestLibVersion());
    }

    /**
     * Test native library extension getter
     */
    public void testNativeLibrary_getExtension() {
        String extension = ".extension";
        TestNativeLibrary testNativeLibrary = new TestNativeLibrary(Collections.emptyMap(), Collections.emptyMap(),
                "", "", extension);
        assertEquals(extension, testNativeLibrary.getExtension());
    }

    /**
     * Test native library compound extension getter
     */
    public void testNativeLibrary_getCompoundExtension() {
        String extension = ".extension";
        TestNativeLibrary testNativeLibrary = new TestNativeLibrary(Collections.emptyMap(), Collections.emptyMap(),
                "", "", extension);
        assertEquals(extension + "c", testNativeLibrary.getCompoundExtension());
    }

    /**
     * Test native library compound extension getter
     */
    public void testNativeLibrary_getMethod() {
        String methodName1 = "test-method-1";
        KNNMethod knnMethod1 = KNNMethod.Builder.builder(MethodComponent.Builder.builder(methodName1).build())
                .build();

        String methodName2 = "test-method-2";
        KNNMethod knnMethod2 = KNNMethod.Builder.builder(MethodComponent.Builder.builder(methodName2).build())
                .build();

        Map<String, KNNMethod> knnMethodMap = ImmutableMap.of(
                methodName1, knnMethod1, methodName2, knnMethod2
        );

        TestNativeLibrary testNativeLibrary = new TestNativeLibrary(knnMethodMap, Collections.emptyMap(),
                "", "", "");
        assertEquals(knnMethod1, testNativeLibrary.getMethod(methodName1));
        assertEquals(knnMethod2, testNativeLibrary.getMethod(methodName2));
        expectThrows(IllegalArgumentException.class, () -> testNativeLibrary.getMethod("invalid"));
    }

    /**
     * Test native library scoring override
     */
    public void testNativeLibrary_score() {
        Map<SpaceType, Function<Float, Float>> translationMap = ImmutableMap.of(SpaceType.L2, s -> s*2);
        TestNativeLibrary testNativeLibrary = new TestNativeLibrary(Collections.emptyMap(), translationMap,
                "", "", "");
        // Test override
        assertEquals(2f, testNativeLibrary.score(1f, SpaceType.L2), 0.0001);

        // Test non-override
        assertEquals(SpaceType.L1.scoreTranslation(1f), testNativeLibrary.score(1f, SpaceType.L1), 0.0001);
    }

    /**
     * Test native library method validation
     */
    public void testNativeLibrary_validateMethod() throws IOException {
        // Invalid - method not supported
        String methodName1 = "test-method-1";
        KNNMethod knnMethod1 = KNNMethod.Builder.builder(MethodComponent.Builder.builder(methodName1).build())
                .build();

        Map<String, KNNMethod> methodMap = ImmutableMap.of(methodName1, knnMethod1);
        TestNativeLibrary testNativeLibrary1 = new TestNativeLibrary(methodMap, Collections.emptyMap(),
                "", "", "");

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder().startObject()
                .field(NAME, "invalid")
                .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext1 = KNNMethodContext.parse(in);
        expectThrows(IllegalArgumentException.class, ()-> testNativeLibrary1.validateMethod(knnMethodContext1));

        // Invalid - method validation
        String methodName2 = "test-method-2";
        KNNMethod knnMethod2 = new KNNMethod(MethodComponent.Builder.builder(methodName2).build(),
                Collections.emptySet()) {
            @Override
            public void validate(KNNMethodContext knnMethodContext) {
                throw new ValidationException();
            }
        };

        methodMap = ImmutableMap.of(methodName2, knnMethod2);
        TestNativeLibrary testNativeLibrary2 = new TestNativeLibrary(methodMap, Collections.emptyMap(),
                "", "", "");
        xContentBuilder = XContentFactory.jsonBuilder().startObject()
                .field(NAME, methodName2)
                .endObject();
        in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext2 = KNNMethodContext.parse(in);
        expectThrows(ValidationException.class, ()-> testNativeLibrary2.validateMethod(knnMethodContext2));
    }

    static class TestNativeLibrary extends KNNLibrary.NativeLibrary {
        /**
         * Constructor for TestNativeLibrary
         *
         * @param methods map of methods the native library supports
         * @param scoreTranslation Map of translation of space type to scores returned by the library
         * @param latestLibraryBuildVersion String representation of latest build version of the library
         * @param latestLibraryVersion String representation of latest version of the library
         * @param extension String representing the extension that library files should use
         */
        public TestNativeLibrary(Map<String, KNNMethod> methods,
                                 Map<SpaceType, Function<Float, Float>> scoreTranslation,
                                 String latestLibraryBuildVersion, String latestLibraryVersion, String extension) {
            super(methods, scoreTranslation, latestLibraryBuildVersion, latestLibraryVersion, extension);
        }
    }
}

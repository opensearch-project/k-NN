/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNMethod;
import org.opensearch.knn.index.SpaceType;

import java.util.Collections;
import java.util.Map;
import java.util.function.Function;

public class NativeLibraryTests extends KNNTestCase {

    /**
     * Test native library extension getter
     */
    public void testGetExtension() {
        String extension = ".extension";
        TestNativeLibrary testNativeLibrary = new TestNativeLibrary(Collections.emptyMap(), Collections.emptyMap(), "", extension);
        assertEquals(extension, testNativeLibrary.getExtension());
    }

    /**
     * Test native library compound extension getter
     */
    public void testGetCompoundExtension() {
        String extension = ".extension";
        TestNativeLibrary testNativeLibrary = new TestNativeLibrary(Collections.emptyMap(), Collections.emptyMap(), "", extension);
        assertEquals(extension + "c", testNativeLibrary.getCompoundExtension());
    }

    /**
     * Test native library scoring override
     */
    public void testScore() {
        Map<SpaceType, Function<Float, Float>> translationMap = ImmutableMap.of(SpaceType.L2, s -> s * 2);
        TestNativeLibrary testNativeLibrary = new TestNativeLibrary(Collections.emptyMap(), translationMap, "", "");
        // Test override
        assertEquals(2f, testNativeLibrary.score(1f, SpaceType.L2), 0.0001);

        // Test non-override
        assertEquals(SpaceType.L1.scoreTranslation(1f), testNativeLibrary.score(1f, SpaceType.L1), 0.0001);
    }

    static class TestNativeLibrary extends NativeLibrary {
        /**
         * Constructor for TestNativeLibrary
         *
         * @param methods map of methods the native library supports
         * @param scoreTranslation Map of translation of space type to scores returned by the library
         * @param currentVersion String representation of current version of the library
         * @param extension String representing the extension that library files should use
         */
        public TestNativeLibrary(
            Map<String, KNNMethod> methods,
            Map<SpaceType, Function<Float, Float>> scoreTranslation,
            String currentVersion,
            String extension
        ) {
            super(methods, Collections.emptyMap(), scoreTranslation, currentVersion, extension);
        }

        @Override
        public Float distanceToRadialThreshold(Float distance, SpaceType spaceType) {
            return 0.0f;
        }

        @Override
        public Float scoreToRadialThreshold(Float score, SpaceType spaceType) {
            return 0.0f;
        }
    }
}

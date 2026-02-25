/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

import org.opensearch.knn.KNNTestCase;

import java.lang.reflect.Method;
import java.lang.reflect.Modifier;

/**
 * Integration tests for KNNLibraryLoader to verify library loading functionality.
 *
 * Tests all non-private methods in KNNLibraryLoader to ensure they can be invoked
 * without throwing exceptions during library loading operations.
 */
public class KNNLibraryLoaderBT extends KNNTestCase {

    /**
     * Tests all non-private methods in KNNLibraryLoader by invoking them.
     *
     * Verifies that library loading methods can be called without exceptions.
     * Uses reflection to discover and test all accessible methods, ensuring
     * comprehensive coverage of the library loading functionality.
     */
    public void testAnnotatedLibraryMethods_whenInvoked_thenLogsResults() {
        Method[] methods = KNNLibraryLoader.class.getDeclaredMethods();

        for (Method method : methods) {
            if (!Modifier.isPrivate(method.getModifiers())) {
                try {
                    method.invoke(null);
                } catch (Exception e) {
                    fail("Library load failed for method " + method.getName() + ": " + e.getMessage());
                }
            }
        }
    }
}

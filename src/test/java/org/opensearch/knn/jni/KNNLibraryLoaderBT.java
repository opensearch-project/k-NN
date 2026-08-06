/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;

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
     * Invokes every no-arg non-private loader method; parameterized loaders are covered by their
     * callers and the direct test below.
     */
    public void testAnnotatedLibraryMethods_whenInvoked_thenLogsResults() {
        Method[] methods = KNNLibraryLoader.class.getDeclaredMethods();

        for (Method method : methods) {
            // Skip parameterized loaders (e.g. the generic loadLibraryByVariant(baseName)); they cannot be
            // invoked blindly and are covered through their no-arg callers (loadFaissLibrary, loadSimdLibrary).
            if (!Modifier.isPrivate(method.getModifiers()) && method.getParameterCount() == 0) {
                try {
                    method.invoke(null);
                } catch (Exception e) {
                    fail("Library load failed for method " + method.getName() + ": " + e.getMessage());
                }
            }
        }
    }

    /**
     * Loads a library by base name through the public variant-selecting entry point.
     */
    public void testLoadLibraryByVariant_whenGivenBaseName_thenLoadsSupportedVariant() {
        try {
            KNNLibraryLoader.loadLibraryByVariant(KNNConstants.FAISS_JNI_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            fail(
                "loadLibraryByVariant failed to resolve and load a variant of "
                    + KNNConstants.FAISS_JNI_LIBRARY_NAME
                    + ": "
                    + e.getMessage()
            );
        }
    }
}

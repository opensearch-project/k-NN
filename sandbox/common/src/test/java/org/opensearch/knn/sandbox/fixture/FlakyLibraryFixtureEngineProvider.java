/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.knn.index.engine.KNNEngineDefinition;
import org.opensearch.knn.index.engine.KNNLibrary;
import org.opensearch.knn.index.engine.NativeEngineService;
import org.opensearch.knn.sandbox.AbstractNativeEngineService;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Its library answers each capability flag and each extension accessor once and throws on every later
 * call. The registry must read them a single time and cache them, so the engine registers and stays
 * usable, including on the segment file path routing that reads extensions.
 */
public final class FlakyLibraryFixtureEngineProvider implements KNNEngineDefinition {

    private static final Map<String, AtomicInteger> CAPABILITY_CALLS = new ConcurrentHashMap<>();

    private static boolean once(String flag, boolean value) {
        if (CAPABILITY_CALLS.computeIfAbsent(flag, k -> new AtomicInteger()).incrementAndGet() > 1) {
            throw new IllegalStateException(flag + " called more than once");
        }
        return value;
    }

    private static final KNNLibrary LIBRARY = new PlainFixtureLibrary(".flakybin") {
        @Override
        public String getExtension() {
            if (CAPABILITY_CALLS.computeIfAbsent("getExtension", k -> new AtomicInteger()).incrementAndGet() > 1) {
                throw new IllegalStateException("getExtension called more than once");
            }
            return ".flakybin";
        }

        @Override
        public String getCompoundExtension() {
            if (CAPABILITY_CALLS.computeIfAbsent("getCompoundExtension", k -> new AtomicInteger()).incrementAndGet() > 1) {
                throw new IllegalStateException("getCompoundExtension called more than once");
            }
            return ".flakybinc";
        }

        @Override
        public boolean supportsIterativeBuild() {
            return once("supportsIterativeBuild", true);
        }

        @Override
        public boolean createsCustomSegmentFiles() {
            return once("createsCustomSegmentFiles", true);
        }

        @Override
        public boolean supportsFilters() {
            return once("supportsFilters", false);
        }

        @Override
        public boolean supportsRadialSearch() {
            // True so the JNIService radial guard has a registered engine that passes it.
            return once("supportsRadialSearch", true);
        }

        @Override
        public boolean supportsNestedFields() {
            return once("supportsNestedFields", false);
        }
    };

    private static final NativeEngineService SERVICE = new AbstractNativeEngineService("flaky-library") {
    };

    @Override
    public String engineName() {
        return "flaky-library";
    }

    @Override
    public KNNLibrary library() {
        return LIBRARY;
    }

    @Override
    public NativeEngineService nativeService() {
        return SERVICE;
    }
}

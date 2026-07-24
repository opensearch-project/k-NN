/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.test.OpenSearchTestCase;

import java.util.List;
import java.util.Map;

import static org.opensearch.knn.sandbox.fixture.FixtureConstants.FIXTURE_ENGINE_NAME;
import static org.opensearch.knn.sandbox.fixture.FixtureConstants.METHOD_PARAMETER_FIXTURE_WINDOW;

/**
 * Exercises the JNIService-layer extension point: every native operation invoked with a registered engine
 * dispatches to that engine's own {@code NativeEngineService} (the fixture's in-memory recorder) with
 * arguments intact, without any engine-identity logic in core — and the built-in branches are untouched.
 * No native library is loaded anywhere in these tests.
 */
public class FixtureJNIServiceDispatchTests extends OpenSearchTestCase {

    private final KNNEngine fixtureEngine = KNNEngine.getEngine(FIXTURE_ENGINE_NAME);
    private final FixtureNativeEngineService fixtureService = FixtureNativeEngineService.INSTANCE;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        fixtureService.reset();
    }

    public void testIndexLifecycleDispatchesToFixtureService() throws Exception {
        final long initHandle = JNIService.initIndex(10, 4, Map.of(), fixtureEngine);
        JNIService.insertToIndex(new int[] { 0, 1, 2 }, 0L, 4, Map.of(), initHandle, fixtureEngine);

        try (Directory directory = new ByteBuffersDirectory()) {
            try (IndexOutput indexOutput = directory.createOutput("fixture_segment", IOContext.DEFAULT)) {
                JNIService.writeIndex(new IndexOutputWithBuffer(indexOutput), initHandle, fixtureEngine, Map.of(), false);
            }
            JNIService.free(initHandle, fixtureEngine, false);

            final long loadHandle;
            try (IndexInput indexInput = directory.openInput("fixture_segment", IOContext.DEFAULT)) {
                loadHandle = JNIService.loadIndex(new IndexInputWithBuffer(indexInput), Map.of(), fixtureEngine);
            }

            final KNNQueryResult[] results = JNIService.queryIndex(
                loadHandle,
                new float[] { 1f, 2f, 3f, 4f },
                3,
                Map.of(METHOD_PARAMETER_FIXTURE_WINDOW, 7),
                fixtureEngine,
                new long[] { 11L, 12L },
                1,
                new int[] { 9 }
            );
            assertEquals(3, results.length);
            assertEquals(0, results[0].getId());
            JNIService.free(loadHandle, fixtureEngine, false);
        }

        final List<String> opLog = fixtureService.opLog();
        assertEquals(7, opLog.size());
        assertTrue(opLog.get(0).startsWith("initIndex(numDocs=10, dim=4)"));
        assertTrue(opLog.get(1).startsWith("insertToIndex(docs=3, dim=4"));
        assertTrue(opLog.get(2).startsWith("writeIndex"));
        assertTrue(opLog.get(3).startsWith("free"));
        assertTrue(opLog.get(4).startsWith("loadIndex"));
        // The engine-specific query parameter and the filter/nested arrays reached the engine's service intact.
        assertTrue(opLog.get(5).startsWith("queryIndex"));
        assertTrue(opLog.get(5).contains(METHOD_PARAMETER_FIXTURE_WINDOW + "=7"));
        assertTrue(opLog.get(5).contains("filteredIds=2, filterIdsType=1, parentIds=1"));
        assertTrue(opLog.get(6).startsWith("free"));
    }

    public void testDispatchIsIsolatedPerEngine() {
        // A call to one registered engine must reach that engine's service only.
        SecondaryFixtureEngineProvider.OP_LOG.clear();
        final KNNEngine secondary = KNNEngine.getEngine(SecondaryFixtureEngineProvider.SECONDARY_FIXTURE_ENGINE_NAME);
        assertEquals(4242L, JNIService.initIndex(10, 4, Map.of(), secondary));
        assertEquals(List.of("initIndex"), SecondaryFixtureEngineProvider.OP_LOG);
        assertTrue(fixtureService.opLog().isEmpty());

        JNIService.initIndex(10, 4, Map.of(), fixtureEngine);
        assertEquals(1, fixtureService.opLog().size());
        assertEquals(List.of("initIndex"), SecondaryFixtureEngineProvider.OP_LOG);
    }

    public void testUnsupportedOperationsAreDeclinedByTheEngineItself() throws Exception {
        // Even declining an operation is the engine's decision: the call must reach the fixture service
        // (visible in its op log) and the UnsupportedOperationException must originate there, not from any
        // engine-identity check in JNIService.
        try (Directory directory = new ByteBuffersDirectory(); IndexOutput indexOutput = directory.createOutput("t", IOContext.DEFAULT)) {
            expectThrows(
                UnsupportedOperationException.class,
                () -> JNIService.createIndexFromTemplate(
                    new int[] { 0 },
                    0L,
                    4,
                    new IndexOutputWithBuffer(indexOutput),
                    new byte[0],
                    Map.of(),
                    fixtureEngine
                )
            );
        }
        expectThrows(
            UnsupportedOperationException.class,
            () -> JNIService.radiusQueryIndex(42L, new float[] { 1f }, 1.0f, Map.of(), fixtureEngine, 10, null, 0, null)
        );
        assertEquals(List.of("createIndexFromTemplate", "radiusQueryIndex"), fixtureService.opLog());
    }

    public void testBuiltInRoutingIsUntouched() {
        // An engine with no native service must take the built-in branches: NMSLIB has no initIndex there,
        // so JNIService itself rejects it — and the fixture service must never see the call.
        expectThrows(IllegalArgumentException.class, () -> JNIService.initIndex(10, 4, Map.of(), KNNEngine.NMSLIB));
        assertTrue(fixtureService.opLog().isEmpty());
    }
}

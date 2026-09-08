/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.index.engine.NativeEngineService;
import org.opensearch.knn.index.engine.NativeIndexBuildParams;
import org.opensearch.knn.index.engine.NativeSearchParams;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Pure-Java, in-memory {@link NativeEngineService} for the fixture engine. The seam under test is
 * {@code JNIService}'s routing — that every native operation invoked with the fixture engine arrives at
 * THIS service with its arguments intact, and never at the built-in Faiss/Nmslib services. Each call is
 * recorded in an op log the tests assert on.
 *
 * <p>Operations a minimal tenant would not support (template builds, radial search) throw
 * {@link UnsupportedOperationException} after logging, mirroring how a real tenant declines them — the
 * tests assert the throw originates here, proving even unsupported-op dispatch is engine-owned.
 */
public final class FixtureNativeEngineService implements NativeEngineService {

    /** The single instance handed to the core through {@link FixtureEngineProvider}; tests reach the op log through it. */
    public static final FixtureNativeEngineService INSTANCE = new FixtureNativeEngineService();

    private final List<String> opLog = Collections.synchronizedList(new ArrayList<>());
    private final AtomicLong nextHandle = new AtomicLong(1000);

    private FixtureNativeEngineService() {}

    /** Snapshot of the recorded operations, in call order. */
    public List<String> opLog() {
        return List.copyOf(opLog);
    }

    /** Clears the recorded operations between tests. */
    public void reset() {
        opLog.clear();
    }

    @Override
    public long initIndex(NativeIndexBuildParams params) {
        final long handle = nextHandle.incrementAndGet();
        opLog.add(String.format(Locale.ROOT, "initIndex(numDocs=%d, dim=%d) -> %d", params.numDocs(), params.dim(), handle));
        return handle;
    }

    @Override
    public void insertToIndex(int[] docs, long vectorsAddress, long indexAddress, NativeIndexBuildParams params) {
        opLog.add(String.format(Locale.ROOT, "insertToIndex(docs=%d, dim=%d, handle=%d)", docs.length, params.dim(), indexAddress));
    }

    @Override
    public void writeIndex(IndexOutputWithBuffer output, long indexAddress, NativeIndexBuildParams params) {
        opLog.add(String.format(Locale.ROOT, "writeIndex(handle=%d)", indexAddress));
    }

    @Override
    public void createIndexFromTemplate(
        int[] ids,
        long vectorsAddress,
        IndexOutputWithBuffer output,
        byte[] templateIndex,
        NativeIndexBuildParams params
    ) {
        opLog.add("createIndexFromTemplate");
        throw new UnsupportedOperationException("Template-based index builds are not supported by the fixture engine");
    }

    @Override
    public long loadIndex(IndexInputWithBuffer readStream, NativeIndexBuildParams params) {
        final long handle = nextHandle.incrementAndGet();
        opLog.add(String.format(Locale.ROOT, "loadIndex() -> %d", handle));
        return handle;
    }

    @Override
    public KNNQueryResult[] queryIndex(long indexPointer, NativeSearchParams params) {
        // The typed read is part of what the dispatch test asserts on, no cast, the key carries the type.
        final Integer typedWindow = params.methodParameters().get(FixtureConstants.FIXTURE_WINDOW);
        opLog.add(
            String.format(
                Locale.ROOT,
                "queryIndex(handle=%d, k=%d, methodParameters=%s, typedWindow=%d, filteredIds=%d, filterIdsType=%d, parentIds=%d)",
                indexPointer,
                params.k(),
                params.methodParameters().raw(),
                typedWindow,
                params.filteredIds() == null ? -1 : params.filteredIds().length,
                params.filterIdsType(),
                params.parentIds() == null ? -1 : params.parentIds().length
            )
        );
        final KNNQueryResult[] results = new KNNQueryResult[params.k()];
        for (int i = 0; i < params.k(); i++) {
            results[i] = new KNNQueryResult(i, 1.0f / (1 + i));
        }
        return results;
    }

    @Override
    public KNNQueryResult[] radiusQueryIndex(long indexPointer, NativeSearchParams params) {
        opLog.add("radiusQueryIndex");
        throw new UnsupportedOperationException("Radial search is not supported by the fixture engine");
    }

    @Override
    public void free(long indexPointer) {
        opLog.add(String.format(Locale.ROOT, "free(handle=%d)", indexPointer));
    }
}

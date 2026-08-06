/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.index.engine.NativeEngineService;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
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
    public long initIndex(long numDocs, int dim, Map<String, Object> parameters) {
        final long handle = nextHandle.incrementAndGet();
        opLog.add(String.format(Locale.ROOT, "initIndex(numDocs=%d, dim=%d) -> %d", numDocs, dim, handle));
        return handle;
    }

    @Override
    public void insertToIndex(int[] docs, long vectorsAddress, int dimension, Map<String, Object> parameters, long indexAddress) {
        opLog.add(String.format(Locale.ROOT, "insertToIndex(docs=%d, dim=%d, handle=%d)", docs.length, dimension, indexAddress));
    }

    @Override
    public void writeIndex(IndexOutputWithBuffer output, long indexAddress, Map<String, Object> parameters, boolean skipFlat) {
        opLog.add(String.format(Locale.ROOT, "writeIndex(handle=%d)", indexAddress));
    }

    @Override
    public void createIndexFromTemplate(
        int[] ids,
        long vectorsAddress,
        int dim,
        IndexOutputWithBuffer output,
        byte[] templateIndex,
        Map<String, Object> parameters
    ) {
        opLog.add("createIndexFromTemplate");
        throw new UnsupportedOperationException("Template-based index builds are not supported by the fixture engine");
    }

    @Override
    public long loadIndex(IndexInputWithBuffer readStream, Map<String, Object> parameters) {
        final long handle = nextHandle.incrementAndGet();
        opLog.add(String.format(Locale.ROOT, "loadIndex() -> %d", handle));
        return handle;
    }

    @Override
    public KNNQueryResult[] queryIndex(
        long indexPointer,
        float[] queryVector,
        int k,
        Map<String, ?> methodParameters,
        long[] filteredIds,
        int filterIdsType,
        int[] parentIds
    ) {
        opLog.add(
            String.format(
                Locale.ROOT,
                "queryIndex(handle=%d, k=%d, methodParameters=%s, filteredIds=%d, filterIdsType=%d, parentIds=%d)",
                indexPointer,
                k,
                methodParameters,
                filteredIds == null ? -1 : filteredIds.length,
                filterIdsType,
                parentIds == null ? -1 : parentIds.length
            )
        );
        final KNNQueryResult[] results = new KNNQueryResult[k];
        for (int i = 0; i < k; i++) {
            results[i] = new KNNQueryResult(i, 1.0f / (1 + i));
        }
        return results;
    }

    @Override
    public KNNQueryResult[] radiusQueryIndex(
        long indexPointer,
        float[] queryVector,
        float radius,
        Map<String, ?> methodParameters,
        int indexMaxResultWindow,
        long[] filteredIds,
        int filterIdsType,
        int[] parentIds
    ) {
        opLog.add("radiusQueryIndex");
        throw new UnsupportedOperationException("Radial search is not supported by the fixture engine");
    }

    @Override
    public void free(long indexPointer, boolean isBinaryIndex) {
        opLog.add(String.format(Locale.ROOT, "free(handle=%d)", indexPointer));
    }
}

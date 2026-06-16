/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;

import java.util.Map;

/**
 * Service-provider interface that contributes a complete experimental engine from an (opt-in)
 * {@code :sandbox} module, discovered at runtime via {@link java.util.ServiceLoader}. It supplies
 * everything {@link KNNEngine#EXPERIMENTAL} needs to behave like any built-in engine:
 *
 * <ul>
 *   <li>{@link #engineName()} — the engine name users type (e.g. {@code "svs"}); also how
 *       {@code KNNEngine.getEngine(name)} resolves to {@code EXPERIMENTAL}.</li>
 *   <li>{@link #library()} — the {@link KNNLibrary} driving method resolution, the file extension (so the
 *       codec writes/reads the tenant's files), validation and scoring.</li>
 *   <li>the native index lifecycle — {@code JNIService} routes every {@code EXPERIMENTAL} op here, to the
 *       tenant's own JNI library (which may embed a different native engine build), fully separate from the
 *       built-in {@code FaissService}/{@code NmslibService}.</li>
 * </ul>
 *
 * <p>Routing is purely by engine: an {@code EXPERIMENTAL} index is created/loaded under this engine and its
 * files carry the tenant's extension, so create, load, query and free all dispatch here uniformly — no
 * per-op routing key is needed. When no provider is bundled (the default build) {@code EXPERIMENTAL} is
 * inert and the plugin is byte-for-byte upstream. Main holds no compile-time reference to the tenant.
 */
public interface SandboxEngineProvider {

    String engineName();

    KNNLibrary library();

    long initIndex(long numDocs, int dim, Map<String, Object> parameters);

    void insertToIndex(int[] docs, long vectorsAddress, int dimension, Map<String, Object> parameters, long indexAddress);

    void writeIndex(IndexOutputWithBuffer output, long indexAddress, Map<String, Object> parameters, boolean skipFlat);

    void createIndexFromTemplate(
        int[] ids,
        long vectorsAddress,
        int dim,
        IndexOutputWithBuffer output,
        byte[] templateIndex,
        Map<String, Object> parameters
    );

    long loadIndex(IndexInputWithBuffer readStream, Map<String, Object> parameters);

    KNNQueryResult[] queryIndex(
        long indexPointer,
        float[] queryVector,
        int k,
        Map<String, ?> methodParameters,
        long[] filteredIds,
        int filterIdsType,
        int[] parentIds
    );

    KNNQueryResult[] radiusQueryIndex(
        long indexPointer,
        float[] queryVector,
        float radius,
        Map<String, ?> methodParameters,
        int indexMaxResultWindow,
        long[] filteredIds,
        int filterIdsType,
        int[] parentIds
    );

    void free(long indexPointer, boolean isBinaryIndex);
}

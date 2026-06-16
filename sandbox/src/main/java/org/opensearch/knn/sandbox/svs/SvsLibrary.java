/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNMethod;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodResolver;
import org.opensearch.knn.index.engine.NativeLibrary;
import org.opensearch.knn.index.engine.ResolvedMethodContext;
import org.opensearch.knn.index.engine.faiss.Faiss;
import org.opensearch.knn.memoryoptsearch.VectorSearcherFactory;

import java.util.Map;

/**
 * The {@link org.opensearch.knn.index.engine.KNNLibrary} for the experimental Intel SVS engine. SVS indices
 * are faiss-format ({@code IndexSVSVamana} is a {@code faiss::Index}), so this library reuses faiss's scoring
 * and only differs where it must:
 * <ul>
 *   <li>its file {@link #getExtension() extension} is {@code .svs} (so the codec writes/reads SVS files
 *       distinctly, and the load path routes them to the SVS engine by extension);</li>
 *   <li>its sole method is {@code svs_vamana} (it does not expose HNSW/IVF);</li>
 *   <li>it provides no memory-optimized searcher, so queries go through the native SVS graph search in
 *       {@code libopensearchknn_svs} rather than reading vectors directly.</li>
 * </ul>
 * This library lives in the sandbox; main reaches it only through {@code KNNEngine.EXPERIMENTAL} via the
 * {@code SandboxEngineProvider} SPI.
 */
public class SvsLibrary extends NativeLibrary {

    // Compatibility version tag baked into the file name; matches the faiss family SVS shares its format with.
    private static final String CURRENT_VERSION = "165";

    private final MethodResolver methodResolver = new SvsMethodResolver();

    public static final SvsLibrary INSTANCE = new SvsLibrary();

    private SvsLibrary() {
        // score translation is delegated to faiss below, so the base map is empty.
        super(
            Map.<String, KNNMethod>of(SVSConstants.METHOD_SVS_VAMANA, new FaissSVSVamanaMethod()),
            Map.of(),
            CURRENT_VERSION,
            SVSConstants.SVS_EXTENSION
        );
    }

    @Override
    public float score(float rawScore, SpaceType spaceType) {
        return Faiss.INSTANCE.score(rawScore, spaceType);
    }

    @Override
    public Float distanceToRadialThreshold(Float distance, SpaceType spaceType) {
        return Faiss.INSTANCE.distanceToRadialThreshold(distance, spaceType);
    }

    @Override
    public Float scoreToRadialThreshold(Float score, SpaceType spaceType) {
        return Faiss.INSTANCE.scoreToRadialThreshold(score, spaceType);
    }

    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        SpaceType spaceType
    ) {
        return methodResolver.resolveMethod(knnMethodContext, knnMethodConfigContext, shouldRequireTraining, spaceType);
    }

    @Override
    public VectorSearcherFactory getVectorSearcherFactory() {
        // No memory-optimized (direct-vector) search for SVS: queries must run through the native SVS graph.
        return null;
    }
}

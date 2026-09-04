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

import java.util.Locale;
import java.util.Map;

/**
 * {@link org.opensearch.knn.index.engine.KNNLibrary} for the experimental Intel SVS engine. SVS indices are
 * faiss-format, so scoring and radial-threshold translation delegate to faiss; the library differs in its
 * {@code .svs} extension, its single {@code svs_vamana} method, and the absence of a memory-optimized searcher.
 */
public class SvsLibrary extends NativeLibrary {

    private static final String CURRENT_VERSION = "166";

    private final MethodResolver methodResolver = new SvsMethodResolver();

    public static final SvsLibrary INSTANCE = new SvsLibrary();

    private SvsLibrary() {
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
        return requirePositiveRadius(Faiss.INSTANCE.distanceToRadialThreshold(distance, spaceType), spaceType);
    }

    @Override
    public Float scoreToRadialThreshold(Float score, SpaceType spaceType) {
        return requirePositiveRadius(Faiss.INSTANCE.scoreToRadialThreshold(score, spaceType), spaceType);
    }

    // The SVS index only accepts a strictly positive faiss-domain radius; reject at query build (400).
    static Float requirePositiveRadius(Float radius, SpaceType spaceType) {
        if (radius != null && radius <= 0) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "The SVS engine does not support radial thresholds that resolve to a non-positive radius "
                        + "(converted radius was %s for space type [%s]); use a stricter max_distance/min_score",
                    radius,
                    spaceType.getValue()
                )
            );
        }
        return radius;
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
        return null;
    }

    @Override
    public boolean supportsIterativeBuild() {
        return true;
    }

    @Override
    public boolean createsCustomSegmentFiles() {
        return true;
    }

    @Override
    public boolean supportsFilters() {
        return true;
    }

    @Override
    public boolean supportsRadialSearch() {
        // Native range_search; non-positive faiss-domain radii are rejected at query build.
        return true;
    }
}

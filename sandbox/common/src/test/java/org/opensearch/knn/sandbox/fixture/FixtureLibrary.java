/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNMethod;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.NativeLibrary;
import org.opensearch.knn.index.engine.ResolvedMethodContext;

import java.util.Map;

import static org.opensearch.knn.sandbox.fixture.FixtureConstants.FIXTURE_EXTENSION;
import static org.opensearch.knn.sandbox.fixture.FixtureConstants.METHOD_FIXTURE;

/**
 * The {@link org.opensearch.knn.index.engine.KNNLibrary} of the test-only fixture engine.
 */
public final class FixtureLibrary extends NativeLibrary {

    public static final FixtureLibrary INSTANCE = new FixtureLibrary();

    private FixtureLibrary() {
        super(Map.<String, KNNMethod>of(METHOD_FIXTURE, new FixtureMethod()), Map.of(), "1", FIXTURE_EXTENSION);
    }

    @Override
    public Float distanceToRadialThreshold(Float distance, SpaceType spaceType) {
        return distance;
    }

    @Override
    public Float scoreToRadialThreshold(Float score, SpaceType spaceType) {
        return score;
    }

    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        SpaceType spaceType
    ) {
        // The fixture is never used in an index mapping; the extension-point tests do not resolve methods.
        throw new UnsupportedOperationException("The test-only fixture engine does not support method resolution");
    }

    @Override
    public boolean supportsIterativeBuild() {
        return true;
    }

    @Override
    public boolean createsCustomSegmentFiles() {
        return true;
    }
}

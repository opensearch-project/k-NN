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

import static org.opensearch.knn.sandbox.fixture.FixtureConstants.METHOD_FIXTURE;

/**
 * A minimal library shared by the hostile registration fixtures. Pure JVM by default and never used
 * beyond registration.
 */
class PlainFixtureLibrary extends NativeLibrary {

    static final PlainFixtureLibrary INSTANCE = new PlainFixtureLibrary();

    PlainFixtureLibrary() {
        this(null);
    }

    PlainFixtureLibrary(String extension) {
        super(Map.<String, KNNMethod>of(METHOD_FIXTURE, new FixtureMethod()), Map.of(), "1", extension);
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
        throw new UnsupportedOperationException("The hostile registration fixtures do not support method resolution");
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngineDefinition;
import org.opensearch.knn.index.engine.KNNLibrary;
import org.opensearch.knn.index.engine.KNNMethod;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.NativeEngineService;
import org.opensearch.knn.index.engine.NativeLibrary;
import org.opensearch.knn.index.engine.ResolvedMethodContext;
import org.opensearch.knn.sandbox.AbstractNativeEngineService;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * A second, pure-JVM-flagged fixture engine used to prove per-engine dispatch isolation: calls to one
 * registered engine must never reach another engine's {@link NativeEngineService}.
 */
public final class SecondaryFixtureEngineProvider implements KNNEngineDefinition {

    public static final String SECONDARY_FIXTURE_ENGINE_NAME = "sandbox_fixture_secondary";

    /** Operations observed by the secondary engine's service, for isolation assertions. */
    public static final List<String> OP_LOG = Collections.synchronizedList(new ArrayList<>());

    static final NativeEngineService SERVICE = new AbstractNativeEngineService(SECONDARY_FIXTURE_ENGINE_NAME) {
        @Override
        public long initIndex(long numDocs, int dim, Map<String, Object> parameters) {
            OP_LOG.add("initIndex");
            return 4242L;
        }
    };

    private static final KNNLibrary LIBRARY = new NativeLibrary(
        Map.<String, KNNMethod>of(FixtureConstants.METHOD_FIXTURE, new FixtureMethod()),
        Map.of(),
        "1",
        ".fixture2bin"
    ) {
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
            throw new UnsupportedOperationException("The secondary fixture engine does not support method resolution");
        }
    };

    @Override
    public String engineName() {
        return SECONDARY_FIXTURE_ENGINE_NAME;
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

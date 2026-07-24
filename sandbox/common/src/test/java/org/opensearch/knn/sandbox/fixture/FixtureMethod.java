/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContextImpl;
import org.opensearch.knn.index.engine.KNNLibrarySearchContext;
import org.opensearch.knn.index.engine.KNNMethod;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.sandbox.fixture.FixtureConstants.METHOD_FIXTURE;

/**
 * The fixture library's single method. Deliberately minimal: L2 only, no training, no encoders.
 */
public final class FixtureMethod implements KNNMethod {

    private final KNNLibrarySearchContext searchContext = new FixtureSearchContext();

    @Override
    public boolean isSpaceTypeSupported(SpaceType space) {
        return space == SpaceType.L2;
    }

    @Override
    public ValidationException validate(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
        return null;
    }

    @Override
    public boolean isTrainingRequired(KNNMethodContext knnMethodContext) {
        return false;
    }

    @Override
    public int estimateOverheadInKB(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
        return 0;
    }

    @Override
    public KNNLibraryIndexingContext getKNNLibraryIndexingContext(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        return KNNLibraryIndexingContextImpl.builder().parameters(Map.of(NAME, METHOD_FIXTURE)).build();
    }

    @Override
    public KNNLibrarySearchContext getKNNLibrarySearchContext() {
        return searchContext;
    }
}

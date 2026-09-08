/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.Version;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.ResolvedIndexSpec;
import org.opensearch.test.OpenSearchTestCase;

import java.util.HashMap;

import static org.opensearch.knn.sandbox.fixture.FixtureConstants.FIXTURE_ENGINE_NAME;
import static org.opensearch.knn.sandbox.fixture.FixtureConstants.METHOD_FIXTURE;

/**
 * The field mapper requires a {@link ResolvedIndexSpec} on every mapped field. A method implementing
 * {@code KNNMethod} directly (this fixture, and real tenants like the BruteForce example) must attach one
 * to its indexing context; these tests pin that contract so a regression fails here instead of as a 500
 * at mapping time.
 */
public class FixtureMethodTests extends OpenSearchTestCase {

    public void testIndexingContextCarriesAResolvedSpec() {
        final KNNEngine fixture = KNNEngine.getEngine(FIXTURE_ENGINE_NAME);
        final KNNMethodContext methodContext = new KNNMethodContext(
            fixture,
            SpaceType.L2,
            new MethodComponentContext(METHOD_FIXTURE, new HashMap<>())
        );
        final KNNMethodConfigContext configContext = KNNMethodConfigContext.builder()
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(4)
            .versionCreated(Version.CURRENT)
            .build();
        final ResolvedIndexSpec spec = new FixtureMethod().getKNNLibraryIndexingContext(methodContext, configContext).getResolvedSpec();
        assertNotNull(spec);
        assertSame(fixture, spec.getEngine());
        assertEquals(METHOD_FIXTURE, spec.getMethodName());
        assertEquals(4, spec.getDimension());
        assertEquals(VectorDataType.FLOAT, spec.getVectorDataType());
    }

    public void testIndexingContextSurvivesNullContexts() {
        final ResolvedIndexSpec spec = new FixtureMethod().getKNNLibraryIndexingContext(null, null).getResolvedSpec();
        assertNotNull(spec);
        assertEquals(METHOD_FIXTURE, spec.getMethodName());
    }
}

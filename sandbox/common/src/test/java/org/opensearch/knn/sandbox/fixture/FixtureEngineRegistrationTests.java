/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.knn.index.VectorQueryType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.model.QueryContext;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Arrays;

import static org.opensearch.knn.sandbox.fixture.FixtureConstants.BAD_FIXTURE_ENGINE_NAME;
import static org.opensearch.knn.sandbox.fixture.FixtureConstants.FIXTURE_ENGINE_NAME;
import static org.opensearch.knn.sandbox.fixture.FixtureConstants.FIXTURE_EXTENSION;
import static org.opensearch.knn.sandbox.fixture.FixtureConstants.METHOD_FIXTURE;
import static org.opensearch.knn.sandbox.fixture.FixtureConstants.METHOD_PARAMETER_FIXTURE_WINDOW;

/**
 * Exercises the KNNEngine-layer extension point: a {@code KNNEngineDefinition} on the classpath (the
 * fixture, registered through META-INF/services in the sandbox TEST resources) becomes a first-class
 * {@code KNNEngine} — resolvable by name, present in {@code values()}, carrying its own native service and
 * capability flags — while the built-in engines are untouched.
 */
public class FixtureEngineRegistrationTests extends OpenSearchTestCase {

    public void testFixtureEngineIsRegisteredByName() {
        final KNNEngine fixture = KNNEngine.getEngine(FIXTURE_ENGINE_NAME);
        assertNotNull(fixture);
        assertEquals(FIXTURE_ENGINE_NAME, fixture.getName());
        // Resolution is case-insensitive, matching the built-in engines' behavior.
        assertSame(fixture, KNNEngine.getEngine(FIXTURE_ENGINE_NAME.toUpperCase(java.util.Locale.ROOT)));
        // The constant-style identifier mirrors the former enum name() contract.
        assertEquals(FIXTURE_ENGINE_NAME.toUpperCase(java.util.Locale.ROOT), fixture.name());
        assertEquals(FIXTURE_ENGINE_NAME.toUpperCase(java.util.Locale.ROOT), fixture.toString());
    }

    public void testFixtureEngineAppearsInValuesExactlyOnce() {
        final long count = Arrays.stream(KNNEngine.values()).filter(e -> FIXTURE_ENGINE_NAME.equals(e.getName())).count();
        assertEquals(1, count);
    }

    public void testValuesListsBuiltInsFirstInDeclarationOrder() {
        final KNNEngine[] values = KNNEngine.values();
        assertSame(KNNEngine.NMSLIB, values[0]);
        assertSame(KNNEngine.FAISS, values[1]);
        assertSame(KNNEngine.LUCENE, values[2]);
        assertSame(KNNEngine.UNDEFINED, values[3]);
    }

    public void testFixtureEngineCarriesItsOwnNativeService() {
        final KNNEngine fixture = KNNEngine.getEngine(FIXTURE_ENGINE_NAME);
        assertSame(FixtureNativeEngineService.INSTANCE, fixture.getNativeService());
    }

    public void testBuiltInEnginesAreUnaffected() {
        // Built-ins resolve exactly as before and carry no native service (their JNI lifecycle is the
        // core FaissService/NmslibService, reached through JNIService's built-in branches).
        assertNull(KNNEngine.FAISS.getNativeService());
        assertNull(KNNEngine.LUCENE.getNativeService());
        assertNull(KNNEngine.NMSLIB.getNativeService());
        assertNull(KNNEngine.UNDEFINED.getNativeService());
        assertSame(KNNEngine.FAISS, KNNEngine.getEngine("faiss"));
        assertSame(KNNEngine.DEFAULT, KNNEngine.FAISS);
    }

    public void testCapabilityFlagsFoldIntoEngineBehavior() {
        final KNNEngine fixture = KNNEngine.getEngine(FIXTURE_ENGINE_NAME);
        assertTrue(fixture.supportsIterativeBuild());
        assertTrue(fixture.createsCustomSegmentFiles());
        assertFalse(fixture.supportsFilters());
        assertFalse(fixture.supportsRadialSearch());
        assertFalse(fixture.supportsNestedFields());
        assertFalse(KNNEngine.ENGINES_SUPPORTING_RADIAL_SEARCH.contains(fixture));
        assertFalse(KNNEngine.ENGINES_SUPPORTING_NESTED_FIELDS.contains(fixture));
        assertTrue(KNNEngine.ENGINES_SUPPORTING_RADIAL_SEARCH.contains(KNNEngine.FAISS));
        assertTrue(KNNEngine.ENGINES_SUPPORTING_NESTED_FIELDS.contains(KNNEngine.LUCENE));
    }

    public void testEngineResolvedFromCustomSegmentFilePath() {
        // createsCustomSegmentFiles() folds the fixture into the custom-segment-file set, which is what
        // getEngineNameFromPath iterates.
        final KNNEngine fixture = KNNEngine.getEngine(FIXTURE_ENGINE_NAME);
        assertSame(fixture, KNNEngine.getEngineNameFromPath("_0_165_target_field" + FIXTURE_EXTENSION));
        assertSame(fixture, KNNEngine.getEngineNameFromPath("_0_165_target_field" + fixture.getCompoundExtension()));
    }

    public void testBrokenDefinitionIsSkippedWithoutPoisoningRegistration() {
        // BadFixtureEngineProvider (registered alongside the fixture) throws from library(); the registry
        // skips it, so registration survives and every other engine still resolves.
        expectThrows(IllegalArgumentException.class, () -> KNNEngine.getEngine(BAD_FIXTURE_ENGINE_NAME));
        assertNotNull(KNNEngine.getEngine(FIXTURE_ENGINE_NAME));
        assertSame(KNNEngine.FAISS, KNNEngine.getEngine("faiss"));
        assertSame(KNNEngine.LUCENE, KNNEngine.getEngine("lucene"));
        assertSame(KNNEngine.NMSLIB, KNNEngine.getEngine("nmslib"));
        assertSame(KNNEngine.UNDEFINED, KNNEngine.getEngine("undefined"));
    }

    public void testEngineExposesItsSearchContext() {
        final KNNEngine fixture = KNNEngine.getEngine(FIXTURE_ENGINE_NAME);
        final QueryContext queryContext = new QueryContext(VectorQueryType.K);
        assertTrue(
            fixture.getKNNLibrarySearchContext(METHOD_FIXTURE)
                .supportedMethodParameters(queryContext)
                .containsKey(METHOD_PARAMETER_FIXTURE_WINDOW)
        );
    }
}

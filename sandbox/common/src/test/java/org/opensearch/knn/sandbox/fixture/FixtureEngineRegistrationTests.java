/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.knn.index.VectorQueryType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.model.QueryContext;
import org.opensearch.knn.index.mapper.FaissFieldStrategy;
import org.opensearch.knn.index.mapper.LuceneFieldStrategy;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Arrays;
import java.util.List;

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

    public void testInitializeIsCalledWithAContext() {
        KNNEngine.getEngine(FIXTURE_ENGINE_NAME); // ensure discovery ran
        assertTrue(FixtureEngineProvider.initialized);
        // Outside a node the context object is EMPTY, never null.
        assertFalse(FixtureEngineProvider.contextWasNull);
    }

    public void testInitializeFailureIsSkipped() {
        // InitThrowsFixtureEngineProvider is well formed but throws from initialize.
        assertTrue(Arrays.stream(KNNEngine.values()).noneMatch(engine -> "init-throws".equals(engine.getName())));
    }

    public void testDuplicateNameClaimantsAreAllDroppedWithoutInitialize() {
        KNNEngine.getEngine(FIXTURE_ENGINE_NAME); // ensure discovery ran
        assertTrue(Arrays.stream(KNNEngine.values()).noneMatch(engine -> "duplicate-name".equals(engine.getName())));
        assertFalse(DuplicateNameFixtureEngineProviderOne.initialized);
        assertFalse(DuplicateNameFixtureEngineProviderTwo.initialized);
    }

    public void testReentrantInitializeSeesBuiltInsAndStillRegisters() {
        // The definition calls KNNEngine.getEngine from its initialize. Discovery must not recurse.
        assertNotNull(KNNEngine.getEngine("reentrant-init"));
        assertTrue(ReentrantInitFixtureEngineProvider.sawFaissDuringInitialize);
    }

    public void testCapabilitiesAreReadOnceAndCached() {
        // The flaky library throws on a second call to any capability flag. The engine registers and
        // answers every flag from the values cached at discovery, asked twice to prove it.
        final KNNEngine flaky = KNNEngine.getEngine("flaky-library");
        for (int i = 0; i < 2; i++) {
            assertTrue(flaky.supportsIterativeBuild());
            assertTrue(flaky.createsCustomSegmentFiles());
            assertFalse(flaky.supportsFilters());
            assertTrue(flaky.supportsRadialSearch());
            assertFalse(flaky.supportsNestedFields());
        }
    }

    public void testFailedInitializeReleasesItsExtensionClaim() {
        // The loser is listed first, claims the shared extension and fails initialize. The winner with the
        // same extension must still register.
        assertTrue(Arrays.stream(KNNEngine.values()).noneMatch(engine -> "shared-extension-loser".equals(engine.getName())));
        assertNotNull(KNNEngine.getEngine("shared-extension-winner"));
    }

    public void testExtensionsAreReadOnceAndCached() {
        // The flaky library throws on a second call to either extension accessor. Path routing answers
        // from the values cached at discovery, asked twice to prove it.
        final KNNEngine flaky = KNNEngine.getEngine("flaky-library");
        for (int i = 0; i < 2; i++) {
            assertEquals(".flakybin", flaky.getExtension());
            assertEquals(".flakybinc", flaky.getCompoundExtension());
        }
        assertSame(flaky, KNNEngine.getEngineNameFromPath("_0_165_field.flakybin"));
    }

    public void testLookupsDoNotWaitOnDiscovery() {
        // The concurrent-lookup fixture probes KNNEngine from another thread inside its own initialize.
        KNNEngine.getEngine(FIXTURE_ENGINE_NAME); // ensure discovery ran
        assertNotNull(KNNEngine.getEngine("concurrent-lookup"));
        assertTrue(ConcurrentLookupFixtureEngineProvider.probeReturnedDuringInitialize);
    }

    public void testCompoundExtensionCollisionIsRejected() {
        // The fixture declares faiss's compound extension. It must be dropped at the claim, before
        // initialize, and faiss must keep resolving its own compound files.
        assertTrue(Arrays.stream(KNNEngine.values()).noneMatch(engine -> "compound-collision".equals(engine.getName())));
        assertFalse(CompoundCollisionFixtureEngineProvider.initialized);
        assertSame(KNNEngine.FAISS, KNNEngine.getEngineNameFromPath("_0_165_field" + KNNEngine.FAISS.getCompoundExtension()));
    }

    public void testProviderLoadFailuresAreSkipped() {
        // The services file starts with a provider whose constructor throws and a provider class that does
        // not exist. Everything after them registering proves discovery survived both failure modes.
        assertNotNull(KNNEngine.getEngine(FIXTURE_ENGINE_NAME));
        assertTrue(Arrays.stream(KNNEngine.values()).noneMatch(engine -> "constructor-throws".equals(engine.getName())));
    }

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
        assertTrue(KNNEngine.FAISS.supportsRadialSearch());
        assertTrue(KNNEngine.LUCENE.supportsNestedFields());
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

    public void testFieldStrategyComesFromTheDefinition() {
        final KNNEngine fixture = KNNEngine.getEngine(FIXTURE_ENGINE_NAME);
        assertSame(FaissFieldStrategy.INSTANCE, fixture.getFieldStrategy());
    }

    public void testFieldStrategyAbsentThrowsWithTheMissingHookNamed() {
        // The secondary fixture declares no field strategy: mapping through it must fail loud, never fall
        // back silently to another engine's strategy.
        final KNNEngine secondary = KNNEngine.getEngine(SecondaryFixtureEngineProvider.SECONDARY_FIXTURE_ENGINE_NAME);
        final UnsupportedOperationException e = expectThrows(UnsupportedOperationException.class, secondary::getFieldStrategy);
        assertTrue(e.getMessage(), e.getMessage().contains("did not provide a fieldStrategy"));
    }

    public void testBuiltInFieldStrategiesAreUnchanged() {
        assertSame(LuceneFieldStrategy.INSTANCE, KNNEngine.LUCENE.getFieldStrategy());
        assertSame(FaissFieldStrategy.INSTANCE, KNNEngine.FAISS.getFieldStrategy());
        assertSame(FaissFieldStrategy.INSTANCE, KNNEngine.NMSLIB.getFieldStrategy());
        expectThrows(UnsupportedOperationException.class, KNNEngine.UNDEFINED::getFieldStrategy);
    }

    /** Guards the close test: the definitions drain on the first close and never repopulate (discovery
     * is once per JVM), so only the first execution in a JVM can observe the full order. */
    private static final java.util.concurrent.atomic.AtomicBoolean CLOSE_TEST_RAN = new java.util.concurrent.atomic.AtomicBoolean();

    public void testCloseRunsInReverseRegistrationOrderAndSurvivesAThrow() {
        assumeTrue("close order is observable only on the first run in a JVM", CLOSE_TEST_RAN.compareAndSet(false, true));
        KNNEngine.getEngine(FIXTURE_ENGINE_NAME); // ensure discovery ran
        FixtureConstants.CLOSE_ORDER.clear();
        KNNEngine.closeEngineDefinitions();
        // Reverse registration order, and the close-throws fixture (which closes first) does not stop the
        // definitions after it.
        assertEquals(
            List.of(
                CloseThrowsFixtureEngineProvider.CLOSE_THROWS_ENGINE_NAME,
                SecondaryFixtureEngineProvider.SECONDARY_FIXTURE_ENGINE_NAME,
                FIXTURE_ENGINE_NAME
            ),
            FixtureConstants.CLOSE_ORDER
        );
        // Idempotent: the definitions are drained on the first call.
        KNNEngine.closeEngineDefinitions();
        assertEquals(3, FixtureConstants.CLOSE_ORDER.size());
        // The engines stay registered, lookups keep working during shutdown.
        assertNotNull(KNNEngine.getEngine(FIXTURE_ENGINE_NAME));
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

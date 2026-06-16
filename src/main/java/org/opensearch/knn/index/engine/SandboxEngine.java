/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.SpaceType;

import java.util.ServiceLoader;

/**
 * Discovers the optional {@link SandboxEngineProvider} (contributed by an opt-in {@code :sandbox} module)
 * once at class load, and exposes what {@link KNNEngine#EXPERIMENTAL} and {@code JNIService} need from it.
 * When no provider is bundled (the default build), {@link #isPresent()} is {@code false}, {@code EXPERIMENTAL}
 * is inert ({@link #library()} returns an {@link InertSandboxLibrary} that never resolves anything), and the
 * plugin behaves exactly as upstream.
 */
public final class SandboxEngine {

    private static final Logger logger = LogManager.getLogger(SandboxEngine.class);

    private static final SandboxEngineProvider PROVIDER = load();
    private static final KNNLibrary INERT_LIBRARY = new InertSandboxLibrary();

    private SandboxEngine() {}

    private static SandboxEngineProvider load() {
        SandboxEngineProvider found = null;
        for (SandboxEngineProvider provider : ServiceLoader.load(SandboxEngineProvider.class, SandboxEngine.class.getClassLoader())) {
            try {
                if (found == null) {
                    found = provider;
                } else {
                    logger.warn("Multiple SandboxEngineProviders found; ignoring [{}]", provider.getClass().getName());
                }
            } catch (Exception | LinkageError e) {
                logger.warn("Skipping misconfigured SandboxEngineProvider", e);
            }
        }
        return found;
    }

    public static boolean isPresent() {
        return PROVIDER != null;
    }

    public static SandboxEngineProvider provider() {
        return PROVIDER;
    }

    /** Engine name to expose for EXPERIMENTAL (the tenant's name, e.g. "svs"; a placeholder when absent). */
    public static String engineName() {
        return PROVIDER != null ? PROVIDER.engineName() : "experimental";
    }

    /** Library backing EXPERIMENTAL; an inert placeholder when no tenant is bundled. */
    public static KNNLibrary library() {
        return PROVIDER != null ? PROVIDER.library() : INERT_LIBRARY;
    }

    /**
     * Placeholder {@link KNNLibrary} behind {@code KNNEngine.EXPERIMENTAL} when no sandbox tenant is
     * bundled (the default build). No field can carry the engine in that case ({@code getEngine} resolves
     * it only when a provider {@link #isPresent()}), so none of these methods is reachable through a real
     * index — but code that iterates all engines (tests, diagnostics) may still touch the enum value.
     * Iteration-safe accessors return benign values; anything implying actual use throws
     * {@link UnsupportedOperationException} instead of NPE-ing on a null library.
     */
    private static final class InertSandboxLibrary implements KNNLibrary {

        private static final String NOT_BUNDLED = "No experimental sandbox engine is bundled in this build";

        @Override
        public String getVersion() {
            return "unavailable";
        }

        @Override
        public String getExtension() {
            // Never written or read: the codec only produces files for fields whose engine resolved to
            // EXPERIMENTAL, which requires a bundled provider; getEngineNameFromPath is isPresent()-guarded.
            return ".sandbox-inert";
        }

        @Override
        public String getCompoundExtension() {
            return ".sandbox-inertc";
        }

        @Override
        public float score(float rawScore, SpaceType spaceType) {
            throw new UnsupportedOperationException(NOT_BUNDLED);
        }

        @Override
        public Float distanceToRadialThreshold(Float distance, SpaceType spaceType) {
            throw new UnsupportedOperationException(NOT_BUNDLED);
        }

        @Override
        public Float scoreToRadialThreshold(Float score, SpaceType spaceType) {
            throw new UnsupportedOperationException(NOT_BUNDLED);
        }

        @Override
        public ValidationException validateMethod(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
            throw new UnsupportedOperationException(NOT_BUNDLED);
        }

        @Override
        public boolean isTrainingRequired(KNNMethodContext knnMethodContext) {
            throw new UnsupportedOperationException(NOT_BUNDLED);
        }

        @Override
        public int estimateOverheadInKB(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
            throw new UnsupportedOperationException(NOT_BUNDLED);
        }

        @Override
        public KNNLibraryIndexingContext getKNNLibraryIndexingContext(
            KNNMethodContext knnMethodContext,
            KNNMethodConfigContext knnMethodConfigContext
        ) {
            throw new UnsupportedOperationException(NOT_BUNDLED);
        }

        @Override
        public KNNLibrarySearchContext getKNNLibrarySearchContext(String methodName) {
            throw new UnsupportedOperationException(NOT_BUNDLED);
        }

        @Override
        public Boolean isInitialized() {
            return false;
        }

        @Override
        public void setInitialized(Boolean isInitialized) {
            // no-op: there is nothing to initialize
        }

        @Override
        public ResolvedMethodContext resolveMethod(
            KNNMethodContext knnMethodContext,
            KNNMethodConfigContext knnMethodConfigContext,
            boolean shouldRequireTraining,
            SpaceType spaceType
        ) {
            throw new UnsupportedOperationException(NOT_BUNDLED);
        }
    }
}

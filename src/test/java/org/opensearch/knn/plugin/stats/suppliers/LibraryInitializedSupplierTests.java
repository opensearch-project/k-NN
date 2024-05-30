/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.plugin.stats.suppliers;

import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.KNNMethod;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNLibrary;
import org.opensearch.knn.training.VectorSpaceInfo;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Map;

public class LibraryInitializedSupplierTests extends OpenSearchTestCase {

    public void testEngineInitialized() {
        KNNLibrary knnLibrary = new TestLibrary();
        LibraryInitializedSupplier libraryInitializedSupplier = new LibraryInitializedSupplier(knnLibrary);
        knnLibrary.setInitialized(false);
        assertFalse(libraryInitializedSupplier.get());
        knnLibrary.setInitialized(true);
        assertTrue(libraryInitializedSupplier.get());
    }

    private static class TestLibrary implements KNNLibrary {
        private Boolean initialized;

        TestLibrary() {
            this.initialized = false;
        }

        @Override
        public String getVersion() {
            return null;
        }

        @Override
        public String getExtension() {
            return null;
        }

        @Override
        public String getCompoundExtension() {
            return null;
        }

        @Override
        public KNNMethod getMethod(String methodName) {
            return null;
        }

        @Override
        public float score(float rawScore, SpaceType spaceType) {
            return 0;
        }

        @Override
        public Float distanceToRadialThreshold(Float distance, SpaceType spaceType) {
            return 0.0f;
        }

        @Override
        public Float scoreToRadialThreshold(Float score, SpaceType spaceType) {
            return 0.0f;
        }

        @Override
        public ValidationException validateMethod(KNNMethodContext knnMethodContext) {
            return null;
        }

        @Override
        public ValidationException validateMethodWithData(KNNMethodContext knnMethodContext, VectorSpaceInfo vectorSpaceInfo) {
            return null;
        }

        @Override
        public boolean isTrainingRequired(KNNMethodContext knnMethodContext) {
            return false;
        }

        @Override
        public int estimateOverheadInKB(KNNMethodContext knnMethodContext, int dimension) {
            return 0;
        }

        @Override
        public Map<String, Object> getMethodAsMap(KNNMethodContext knnMethodContext) {
            return null;
        }

        @Override
        public Boolean isInitialized() {
            return initialized;
        }

        @Override
        public void setInitialized(Boolean isInitialized) {
            this.initialized = isInitialized;
        }
    }
}

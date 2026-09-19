/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.engine.KNNEngine;

public class NativeEngines990HalfFloatKnnVectorsFormatTests extends KNNTestCase {

    public void testDefaultConstructor_thenSucceeds() {
        NativeEngines990HalfFloatKnnVectorsFormat format = new NativeEngines990HalfFloatKnnVectorsFormat();
        assertNotNull(format);
        assertTrue(format.toString().contains("KNN1040HalfFloatFlatVectorsFormat"));
    }

    public void testConstructor_withApproximateThresholdAndFactory_thenSucceeds() {
        NativeEngines990HalfFloatKnnVectorsFormat format = new NativeEngines990HalfFloatKnnVectorsFormat(
            100,
            new NativeIndexBuildStrategyFactory()
        );
        assertNotNull(format);
        assertTrue(format.toString().contains("approximateThreshold=100"));
        assertTrue(format.toString().contains("KNN1040HalfFloatFlatVectorsFormat"));
    }

    public void testGetName_returnsClassName() {
        assertEquals("NativeEngines990HalfFloatKnnVectorsFormat", new NativeEngines990HalfFloatKnnVectorsFormat().getName());
    }

    public void testGetMaxDimensions_whenCalled_thenUseFaissEngine() {
        try (MockedStatic<KNNEngine> mockedKNNEngine = Mockito.mockStatic(KNNEngine.class)) {
            mockedKNNEngine.when(() -> KNNEngine.getMaxDimensionByEngine(KNNEngine.FAISS)).thenReturn(16000);

            NativeEngines990HalfFloatKnnVectorsFormat format = new NativeEngines990HalfFloatKnnVectorsFormat();
            int result = format.getMaxDimensions("test-field");

            assertEquals(16000, result);
            mockedKNNEngine.verify(() -> KNNEngine.getMaxDimensionByEngine(KNNEngine.FAISS));
        }
    }

    // --- Coverage: distinct SPI name, required for the half-float format to be resolvable ---

    public void testGetName_whenHalfFloatFormat_thenDistinctFromFloatFormat() {
        String floatName = new NativeEngines990KnnVectorsFormat().getName();
        String halfFloatName = new NativeEngines990HalfFloatKnnVectorsFormat().getName();
        assertEquals("NativeEngines990KnnVectorsFormat", floatName);
        assertEquals("NativeEngines990HalfFloatKnnVectorsFormat", halfFloatName);
        assertNotEquals("SPI names must differ or the codec cannot resolve both formats", floatName, halfFloatName);
    }

    public void testHalfFloatFormat_thenUsesFp16Delegate() {
        assertTrue(new NativeEngines990HalfFloatKnnVectorsFormat().toString().contains("KNN1040HalfFloatFlatVectorsFormat"));
    }
}

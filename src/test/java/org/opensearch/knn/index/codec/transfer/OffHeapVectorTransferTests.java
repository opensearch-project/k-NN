/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer;

import lombok.SneakyThrows;
import org.mockito.MockedStatic;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;

import java.util.List;

import static org.mockito.Mockito.mockStatic;

public class OffHeapVectorTransferTests extends KNNTestCase {

    @SneakyThrows
    public void testFloatTransfer() {
        List<float[]> vectors = List.of(
            new float[] { 0.1f, 0.2f },
            new float[] { 0.2f, 0.3f },
            new float[] { 0.3f, 0.4f },
            new float[] { 0.3f, 0.4f },
            new float[] { 0.3f, 0.4f }
        );

        try (MockedStatic<KNNSettings> mockedKNNSettings = mockStatic(KNNSettings.class)) {
            mockedKNNSettings.when(KNNSettings::getVectorStreamingMemoryLimit).thenReturn(new ByteSizeValue(16));

            OffHeapFloatVectorTransfer vectorTransfer = new OffHeapFloatVectorTransfer(8, 5);
            long vectorAddress = 0;
            assertFalse(vectorTransfer.transfer(vectors.get(0), false));
            assertEquals(0, vectorTransfer.getVectorAddress());
            assertTrue(vectorTransfer.transfer(vectors.get(1), false));
            vectorAddress = vectorTransfer.getVectorAddress();
            assertFalse(vectorTransfer.transfer(vectors.get(2), false));
            assertEquals(vectorAddress, vectorTransfer.getVectorAddress());
            assertTrue(vectorTransfer.transfer(vectors.get(3), false));
            assertEquals(vectorAddress, vectorTransfer.getVectorAddress());
            assertFalse(vectorTransfer.transfer(vectors.get(4), false));
            assertTrue(vectorTransfer.flush(false));
            vectorTransfer.reset();
            assertEquals(0, vectorTransfer.getVectorAddress());
            vectorTransfer.close();

        }

    }

    @SneakyThrows
    public void testByteTransfer() {
        List<byte[]> vectors = List.of(
            new byte[] { 0, 1 },
            new byte[] { 2, 3 },
            new byte[] { 4, 5 },
            new byte[] { 6, 7 },
            new byte[] { 8, 9 }
        );

        try (MockedStatic<KNNSettings> mockedKNNSettings = mockStatic(KNNSettings.class)) {
            mockedKNNSettings.when(KNNSettings::getVectorStreamingMemoryLimit).thenReturn(new ByteSizeValue(4));
            OffHeapByteVectorTransfer vectorTransfer = new OffHeapByteVectorTransfer(2, 5);
            long vectorAddress = 0;
            assertFalse(vectorTransfer.transfer(vectors.get(0), false));
            assertEquals(0, vectorTransfer.getVectorAddress());
            assertTrue(vectorTransfer.transfer(vectors.get(1), false));
            vectorAddress = vectorTransfer.getVectorAddress();
            assertFalse(vectorTransfer.transfer(vectors.get(2), false));
            assertEquals(vectorAddress, vectorTransfer.getVectorAddress());
            assertTrue(vectorTransfer.transfer(vectors.get(3), false));
            assertEquals(vectorAddress, vectorTransfer.getVectorAddress());
            assertFalse(vectorTransfer.transfer(vectors.get(4), false));
            assertTrue(vectorTransfer.flush(false));
            vectorTransfer.close();
            assertEquals(0, vectorTransfer.getVectorAddress());
        }
    }

    @SneakyThrows
    public void testBinaryTransfer() {
        List<byte[]> vectors = List.of(
            new byte[] { 0, 1 },
            new byte[] { 2, 3 },
            new byte[] { 4, 5 },
            new byte[] { 6, 7 },
            new byte[] { 8, 9 }
        );

        try (MockedStatic<KNNSettings> mockedKNNSettings = mockStatic(KNNSettings.class)) {
            mockedKNNSettings.when(KNNSettings::getVectorStreamingMemoryLimit).thenReturn(new ByteSizeValue(4));
            OffHeapBinaryVectorTransfer vectorTransfer = new OffHeapBinaryVectorTransfer(2, 5);
            long vectorAddress = 0;
            assertFalse(vectorTransfer.transfer(vectors.get(0), false));
            assertEquals(0, vectorTransfer.getVectorAddress());
            assertTrue(vectorTransfer.transfer(vectors.get(1), false));
            vectorAddress = vectorTransfer.getVectorAddress();
            assertFalse(vectorTransfer.transfer(vectors.get(2), false));
            assertEquals(vectorAddress, vectorTransfer.getVectorAddress());
            assertTrue(vectorTransfer.transfer(vectors.get(3), false));
            assertEquals(vectorAddress, vectorTransfer.getVectorAddress());
            assertFalse(vectorTransfer.transfer(vectors.get(4), false));
            assertTrue(vectorTransfer.flush(false));
            vectorTransfer.close();
        }
    }
}

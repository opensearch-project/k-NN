/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.exactsearch;

import junit.framework.TestCase;
import lombok.SneakyThrows;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.mockito.stubbing.OngoingStubbing;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.vectorvalues.KNNByteVectorValues;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class ByteVectorIdsExactKNNIteratorTests extends TestCase {
    @SneakyThrows
    public void testNextDoc_whenCalled_IterateAllDocs() {
        final SpaceType spaceType = SpaceType.L2;
        final byte[] byteQueryVector = { 1, 2, 3 };
        final float[] queryVector = { 1f, 2f, 3f };
        final int[] filterIds = { 1, 2, 3 };
        final List<byte[]> dataVectors = Arrays.asList(new byte[] { 11, 12, 13 }, new byte[] { 14, 15, 16 }, new byte[] { 17, 18, 19 });
        final List<Float> expectedScores = dataVectors.stream()
            .map(vector -> spaceType.getKnnVectorSimilarityFunction().compare(byteQueryVector, vector))
            .collect(Collectors.toList());

        KNNByteVectorValues values = mock(KNNByteVectorValues.class);
        when(values.getVector()).thenReturn(dataVectors.get(0), dataVectors.get(1), dataVectors.get(2));

        FixedBitSet filterBitSet = new FixedBitSet(4);
        for (int id : filterIds) {
            when(values.advance(id)).thenReturn(id);
            filterBitSet.set(id);
        }

        // Execute and verify
        ByteVectorIdsExactKNNIterator iterator = new ByteVectorIdsExactKNNIterator(
            new BitSetIterator(filterBitSet, filterBitSet.length()),
            queryVector,
            values,
            spaceType
        );
        for (int i = 0; i < filterIds.length; i++) {
            assertEquals(filterIds[i], iterator.nextDoc());
            assertEquals(expectedScores.get(i), (Float) iterator.score());
        }
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    @SneakyThrows
    public void testNextDoc_whenCalled_thenIterateAllDocsWithoutFilter() throws IOException {
        final SpaceType spaceType = SpaceType.L2;
        final byte[] byteQueryVector = { 1, 2, 3 };
        final float[] queryVector = { 1.0f, 2.0f, 3.0f };
        final List<byte[]> dataVectors = Arrays.asList(
            new byte[] { 11, 12, 13 },
            new byte[] { 14, 15, 16 },
            new byte[] { 17, 18, 19 },
            new byte[] { 20, 21, 22 },
            new byte[] { 23, 24, 25 }
        );
        final List<Float> expectedScores = dataVectors.stream()
            .map(vector -> spaceType.getKnnVectorSimilarityFunction().compare(byteQueryVector, vector))
            .collect(Collectors.toList());

        KNNByteVectorValues values = mock(KNNByteVectorValues.class);
        when(values.getVector()).thenReturn(
            dataVectors.get(0),
            dataVectors.get(1),
            dataVectors.get(2),
            dataVectors.get(3),
            dataVectors.get(4)
        );

        // stub return value when nextDoc is called
        OngoingStubbing<Integer> stubbing = when(values.nextDoc());
        for (int i = 0; i < dataVectors.size(); i++) {
            stubbing = stubbing.thenReturn(i);
        }
        // set last return to be Integer.MAX_VALUE to represent no more docs
        stubbing.thenReturn(Integer.MAX_VALUE);

        // Execute and verify
        ByteVectorIdsExactKNNIterator iterator = new ByteVectorIdsExactKNNIterator(queryVector, values, spaceType);
        for (int i = 0; i < dataVectors.size(); i++) {
            assertEquals(i, iterator.nextDoc());
            assertEquals(expectedScores.get(i), iterator.score());
        }
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
        verify(values, never()).advance(anyInt());
    }
}

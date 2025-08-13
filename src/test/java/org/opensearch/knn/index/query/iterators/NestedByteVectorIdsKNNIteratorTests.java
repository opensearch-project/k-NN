/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.iterators;

import junit.framework.TestCase;
import lombok.SneakyThrows;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.vectorvalues.KNNByteVectorValues;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class NestedByteVectorIdsKNNIteratorTests extends TestCase {
    @SneakyThrows
    public void testNextDoc_whenIterate_ReturnBestChildDocsPerParent() {
        final SpaceType spaceType = SpaceType.L2;
        final byte[] byteQueryVector = { 1, 2, 3 };
        final float[] queryVector = { 1.0f, 2.0f, 3.0f };
        final int[] filterIds = { 0, 2, 3 };
        // Parent id for 0 -> 1
        // Parent id for 2, 3 -> 4
        // In bit representation, it is 10010. In long, it is 18.
        final BitSet parentBitSet = new FixedBitSet(new long[] { 18 }, 5);
        final List<byte[]> dataVectors = Arrays.asList(new byte[] { 11, 12, 13 }, new byte[] { 17, 18, 19 }, new byte[] { 14, 15, 16 });
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
        NestedByteVectorIdsKNNIterator iterator = new NestedByteVectorIdsKNNIterator(
            new BitSetIterator(filterBitSet, filterBitSet.length()),
            queryVector,
            values,
            spaceType,
            parentBitSet,
            false
        );
        assertEquals(filterIds[0], iterator.nextDoc());
        assertEquals(expectedScores.get(0), iterator.score());
        assertEquals(filterIds[2], iterator.nextDoc());
        assertEquals(expectedScores.get(2), iterator.score());
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    @SneakyThrows
    public void testNextDoc_whenIterateWithoutFilters_thenReturnBestChildDocsPerParent() {
        final SpaceType spaceType = SpaceType.L2;
        final byte[] byteQueryVector = { 1, 2, 3 };
        final float[] queryVector = { 1.0f, 2.0f, 3.0f };
        // Parent id for 0 -> 1
        // Parent id for 2, 3 -> 4
        // In bit representation, it is 10010. In long, it is 18.
        final BitSet parentBitSet = new FixedBitSet(new long[] { 18 }, 5);
        final List<byte[]> dataVectors = Arrays.asList(new byte[] { 11, 12, 13 }, new byte[] { 17, 18, 19 }, new byte[] { 14, 15, 16 });
        final List<Float> expectedScores = dataVectors.stream()
            .map(vector -> spaceType.getKnnVectorSimilarityFunction().compare(byteQueryVector, vector))
            .collect(Collectors.toList());

        KNNByteVectorValues values = mock(KNNByteVectorValues.class);
        when(values.getVector()).thenReturn(dataVectors.get(0), dataVectors.get(1), dataVectors.get(2));
        when(values.nextDoc()).thenReturn(0, 2, 3, Integer.MAX_VALUE);

        // Execute and verify
        NestedByteVectorIdsKNNIterator iterator = new NestedByteVectorIdsKNNIterator(queryVector, values, spaceType, parentBitSet);
        assertEquals(0, iterator.nextDoc());
        assertEquals(expectedScores.get(0), iterator.score());
        assertEquals(3, iterator.nextDoc());
        assertEquals(expectedScores.get(2), iterator.score());
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
        verify(values, never()).advance(anyInt());
    }

    @SneakyThrows
    public void testNextDoc_whenIterateExpandNested_thenReturnAllChildDocsPerParent() {
        final SpaceType spaceType = SpaceType.L2;
        final byte[] byteQueryVector = { 1, 2, 3 };
        final float[] queryVector = { 1.0f, 2.0f, 3.0f };
        final int[] filterIds = { 0, 2, 3 };
        // Parent id for 0 -> 1
        // Parent id for 2, 3 -> 4
        // In bit representation, it is 10010. In long, it is 18.
        final BitSet parentBitSet = new FixedBitSet(new long[] { 18 }, 5);
        final List<byte[]> dataVectors = Arrays.asList(new byte[] { 11, 12, 13 }, new byte[] { 17, 18, 19 }, new byte[] { 14, 15, 16 });
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
        NestedByteVectorIdsKNNIterator iterator = new NestedByteVectorIdsKNNIterator(
            new BitSetIterator(filterBitSet, filterBitSet.length()),
            queryVector,
            values,
            spaceType,
            parentBitSet,
            true
        );
        assertEquals(filterIds[0], iterator.nextDoc());
        assertEquals(expectedScores.get(0), iterator.score());
        assertEquals(filterIds[1], iterator.nextDoc());
        assertEquals(expectedScores.get(1), iterator.score());
        assertEquals(filterIds[2], iterator.nextDoc());
        assertEquals(expectedScores.get(2), iterator.score());
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }
}

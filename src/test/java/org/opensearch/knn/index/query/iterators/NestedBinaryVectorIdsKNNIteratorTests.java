/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.iterators;

import junit.framework.TestCase;
import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.vectorvalues.KNNBinaryVectorValues;

import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@Log4j2
public class NestedBinaryVectorIdsKNNIteratorTests extends TestCase {
    @SneakyThrows
    public void testNextDoc_whenIterate_ReturnBestChildDocsPerParent() {
        final SpaceType spaceType = SpaceType.HAMMING;
        final byte[] queryVector = { 1, 2, 3 };
        final int[] filterIds = { 0, 2, 3 };
        // Parent id for 0 -> 1
        // Parent id for 2, 3 -> 4
        // In bit representation, it is 10010. In long, it is 18.
        final BitSet parentBitSet = new FixedBitSet(new long[] { 18 }, 5);
        final Map<Integer, byte[]> dataVectors = Map.of(
            0,
            new byte[] { 11, 12, 13 },
            2,
            new byte[] { 14, 15, 16 },
            3,
            new byte[] { 17, 18, 19 }
        );
        final Map<Integer, Float> expectedScores = dataVectors.entrySet()
            .stream()
            .collect(
                Collectors.toMap(Map.Entry::getKey, e -> spaceType.getKnnVectorSimilarityFunction().compare(queryVector, e.getValue()))
            );

        KNNBinaryVectorValues values = mock(KNNBinaryVectorValues.class);
        FixedBitSet filterBitSet = new FixedBitSet(4);
        AtomicInteger lastReturned = new AtomicInteger(-1);
        for (int id : filterIds) {
            when(values.advance(id)).thenAnswer(inv -> {
                int target = inv.getArgument(0);
                lastReturned.set(target);
                return target;
            });
            filterBitSet.set(id);
        }
        when(values.getVector()).thenAnswer(inv -> dataVectors.get(lastReturned.get()));

        // Execute and verify
        NestedBinaryVectorIdsKNNIterator iterator = new NestedBinaryVectorIdsKNNIterator(
            new BitSetIterator(filterBitSet, filterBitSet.length()),
            queryVector,
            values,
            spaceType,
            parentBitSet
        );
        assertEquals(filterIds[0], iterator.nextDoc());
        assertEquals(expectedScores.get(filterIds[0]), iterator.score());
        assertEquals(filterIds[2], iterator.nextDoc());
        assertEquals(expectedScores.get(filterIds[2]), iterator.score());
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    @SneakyThrows
    public void testNextDoc_whenIterateWithoutFilters_thenReturnBestChildDocsPerParent() {
        final SpaceType spaceType = SpaceType.HAMMING;
        final byte[] queryVector = { 1, 2, 3 };
        // Parent id for 0 -> 1
        // Parent id for 2, 3 -> 4
        // In bit representation, it is 10010. In long, it is 18.
        final BitSet parentBitSet = new FixedBitSet(new long[] { 18 }, 5);
        final Map<Integer, byte[]> dataVectors = Map.of(
            0,
            new byte[] { 11, 12, 13 },
            2,
            new byte[] { 14, 15, 16 },
            3,
            new byte[] { 17, 18, 19 }
        );
        final Map<Integer, Float> expectedScores = dataVectors.entrySet()
            .stream()
            .collect(
                Collectors.toMap(Map.Entry::getKey, e -> spaceType.getKnnVectorSimilarityFunction().compare(queryVector, e.getValue()))
            );

        KNNBinaryVectorValues values = mock(KNNBinaryVectorValues.class);
        AtomicInteger lastReturned = new AtomicInteger(-1);
        when(values.advance(anyInt())).thenAnswer(invocation -> {
            int target = invocation.getArgument(0);
            int prev = lastReturned.get();
            assertTrue(prev < target);
            int[] docs = { 0, 2, 3, Integer.MAX_VALUE };
            for (int doc : docs) {
                if (doc >= target) {
                    lastReturned.set(doc);
                    return doc;
                }
            }
            lastReturned.set(Integer.MAX_VALUE);
            return Integer.MAX_VALUE;
        });
        when(values.getVector()).thenAnswer(inv -> dataVectors.get(lastReturned.get()));

        // Execute and verify
        NestedBinaryVectorIdsKNNIterator iterator = new NestedBinaryVectorIdsKNNIterator(
            DocIdSetIterator.range(0, 5),
            queryVector,
            values,
            spaceType,
            parentBitSet
        );
        assertEquals(0, iterator.nextDoc());
        assertEquals(expectedScores.get(0), iterator.score());
        assertEquals(3, iterator.nextDoc());
        assertEquals(expectedScores.get(3), iterator.score());
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }
}

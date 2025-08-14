/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.iterators;

import junit.framework.TestCase;
import lombok.SneakyThrows;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.vectorvalues.KNNBinaryVectorValues;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class BinaryVectorIdsKNNIteratorTests extends TestCase {
    @SneakyThrows
    public void testNextDoc_whenCalled_IterateAllDocs() {
        final SpaceType spaceType = SpaceType.HAMMING;
        final byte[] queryVector = { 1, 2, 3 };
        final int[] filterIds = { 0, 2, 3 };
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
        BinaryVectorIdsKNNIterator iterator = new BinaryVectorIdsKNNIterator(
            new BitSetIterator(filterBitSet, filterBitSet.length()),
            queryVector,
            values,
            spaceType
        );
        for (int filterId : filterIds) {
            assertEquals(filterId, iterator.nextDoc());
            assertEquals(expectedScores.get(filterId), iterator.score());
        }
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    @SneakyThrows
    public void testNextDoc_whenCalled_thenIterateAllDocsWithoutFilter() throws IOException {
        final SpaceType spaceType = SpaceType.HAMMING;
        final byte[] queryVector = { 1, 2, 3 };
        int[] docs = { 0, 1, 2, 3, 4, Integer.MAX_VALUE };
        final List<byte[]> dataVectors = Arrays.asList(
            new byte[] { 11, 12, 13 },
            new byte[] { 14, 15, 16 },
            new byte[] { 17, 18, 19 },
            new byte[] { 20, 21, 22 },
            new byte[] { 23, 24, 25 }
        );
        final List<Float> expectedScores = dataVectors.stream()
            .map(vector -> spaceType.getKnnVectorSimilarityFunction().compare(queryVector, vector))
            .toList();

        KNNBinaryVectorValues values = mock(KNNBinaryVectorValues.class);
        AtomicInteger lastReturned = new AtomicInteger(-1);
        when(values.advance(anyInt())).thenAnswer(invocation -> {
            int target = invocation.getArgument(0);
            int prev = lastReturned.get();
            assertTrue(prev < target);
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
        BinaryVectorIdsKNNIterator iterator = new BinaryVectorIdsKNNIterator(DocIdSetIterator.range(0, 5), queryVector, values, spaceType);
        for (int i = 0; i < dataVectors.size(); i++) {
            assertEquals(i, iterator.nextDoc());
            assertEquals(expectedScores.get(i), iterator.score());
        }
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.iterators;

import lombok.SneakyThrows;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class VectorIdsKNNIteratorTests extends KNNTestCase {
    @SneakyThrows
    public void testNextDoc_whenCalledWithFilters_thenIterateAllDocs() {
        final SpaceType spaceType = SpaceType.L2;
        final float[] queryVector = { 1.0f, 2.0f, 3.0f };
        final int[] filterIds = { 0, 2, 3 };
        final Map<Integer, float[]> dataVectors = Map.of(
            0,
            new float[] { 11.0f, 12.0f, 13.0f },
            2,
            new float[] { 17.0f, 18.0f, 19.0f },
            3,
            new float[] { 14.0f, 15.0f, 16.0f }
        );
        final Map<Integer, Float> expectedScores = dataVectors.entrySet()
            .stream()
            .collect(
                Collectors.toMap(Map.Entry::getKey, e -> spaceType.getKnnVectorSimilarityFunction().compare(queryVector, e.getValue()))
            );

        KNNFloatVectorValues values = mock(KNNFloatVectorValues.class);
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
        VectorIdsKNNIterator iterator = new VectorIdsKNNIterator(
            new BitSetIterator(filterBitSet, filterBitSet.length()),
            queryVector,
            values,
            spaceType
        );
        for (int filterId : filterIds) {
            assertEquals(filterId, iterator.nextDoc());
            assertEquals(expectedScores.get(filterId), (Float) iterator.score());
        }
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    @SneakyThrows
    public void testNextDoc_whenCalledWithoutFilters_thenIterateAllDocs() {
        final SpaceType spaceType = SpaceType.L2;
        final float[] queryVector = { 1.0f, 2.0f, 3.0f };
        int[] docs = { 0, 1, 2, 3, 4, Integer.MAX_VALUE };
        final List<float[]> dataVectors = Arrays.asList(
            new float[] { 11.0f, 12.0f, 13.0f },
            new float[] { 14.0f, 15.0f, 16.0f },
            new float[] { 17.0f, 18.0f, 19.0f },
            new float[] { 20.0f, 21.0f, 22.0f },
            new float[] { 23.0f, 24.0f, 25.0f }
        );
        final List<Float> expectedScores = dataVectors.stream()
            .map(vector -> spaceType.getKnnVectorSimilarityFunction().compare(queryVector, vector))
            .toList();

        KNNFloatVectorValues values = mock(KNNFloatVectorValues.class);
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
        VectorIdsKNNIterator iterator = new VectorIdsKNNIterator(DocIdSetIterator.range(0, 5), queryVector, values, spaceType);
        for (int i = 0; i < dataVectors.size(); i++) {
            assertEquals(i, iterator.nextDoc());
            assertEquals(expectedScores.get(i), (Float) iterator.score());
        }
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }
}

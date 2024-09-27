/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.iterators;

import lombok.SneakyThrows;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class VectorIdsKNNIteratorTests extends KNNTestCase {
    @SneakyThrows
    public void testNextDoc_whenCalled_IterateAllDocs() {
        final SpaceType spaceType = SpaceType.L2;
        final float[] queryVector = { 1.0f, 2.0f, 3.0f };
        final int[] filterIds = { 1, 2, 3 };
        final List<float[]> dataVectors = Arrays.asList(
            new float[] { 11.0f, 12.0f, 13.0f },
            new float[] { 14.0f, 15.0f, 16.0f },
            new float[] { 17.0f, 18.0f, 19.0f }
        );
        final List<Float> expectedScores = dataVectors.stream()
            .map(vector -> spaceType.getKnnVectorSimilarityFunction().compare(queryVector, vector))
            .collect(Collectors.toList());

        KNNFloatVectorValues values = mock(KNNFloatVectorValues.class);
        when(values.getVector()).thenReturn(dataVectors.get(0), dataVectors.get(1), dataVectors.get(2));

        FixedBitSet filterBitSet = new FixedBitSet(4);
        for (int id : filterIds) {
            when(values.advance(id)).thenReturn(id);
            filterBitSet.set(id);
        }

        // Execute and verify
        VectorIdsKNNIterator iterator = new VectorIdsKNNIterator(filterBitSet, queryVector, values, spaceType);
        for (int i = 0; i < filterIds.length; i++) {
            assertEquals(filterIds[i], iterator.nextDoc());
            assertEquals(expectedScores.get(i), (Float) iterator.score());
        }
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }
}

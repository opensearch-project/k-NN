/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.filtered;

import lombok.SneakyThrows;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.util.KNNVectorAsArraySerializer;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class FilteredIdsKNNIteratorTests extends KNNTestCase {
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

        BinaryDocValues values = mock(BinaryDocValues.class);
        final List<BytesRef> byteRefs = dataVectors.stream()
            .map(vector -> new BytesRef(new KNNVectorAsArraySerializer().floatToByteArray(vector)))
            .collect(Collectors.toList());
        when(values.binaryValue()).thenReturn(byteRefs.get(0), byteRefs.get(1), byteRefs.get(2));

        FixedBitSet filterBitSet = new FixedBitSet(4);
        for (int id : filterIds) {
            when(values.advance(id)).thenReturn(id);
            filterBitSet.set(id);
        }

        // Execute and verify
        FilteredIdsKNNIterator iterator = new FilteredIdsKNNIterator(filterBitSet, queryVector, values, spaceType);
        for (int i = 0; i < filterIds.length; i++) {
            assertEquals(filterIds[i], iterator.nextDoc());
            assertEquals(expectedScores.get(i), (Float) iterator.score());
        }
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }
}

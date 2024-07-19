/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.filtered;

import junit.framework.TestCase;
import lombok.SneakyThrows;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.index.SpaceType;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class FilteredIdsKNNByteIteratorTests extends TestCase {
    @SneakyThrows
    public void testNextDoc_whenCalled_IterateAllDocs() {
        final SpaceType spaceType = SpaceType.HAMMING;
        final byte[] queryVector = { 1, 2, 3 };
        final int[] filterIds = { 1, 2, 3 };
        final List<byte[]> dataVectors = Arrays.asList(new byte[] { 11, 12, 13 }, new byte[] { 14, 15, 16 }, new byte[] { 17, 18, 19 });
        final List<Float> expectedScores = dataVectors.stream()
            .map(vector -> spaceType.getKnnVectorSimilarityFunction().compare(queryVector, vector))
            .collect(Collectors.toList());

        BinaryDocValues values = mock(BinaryDocValues.class);
        final List<BytesRef> byteRefs = dataVectors.stream().map(vector -> new BytesRef(vector)).collect(Collectors.toList());
        when(values.binaryValue()).thenReturn(byteRefs.get(0), byteRefs.get(1), byteRefs.get(2));

        FixedBitSet filterBitSet = new FixedBitSet(4);
        for (int id : filterIds) {
            when(values.advance(id)).thenReturn(id);
            filterBitSet.set(id);
        }

        // Execute and verify
        FilteredIdsKNNByteIterator iterator = new FilteredIdsKNNByteIterator(filterBitSet, queryVector, values, spaceType);
        for (int i = 0; i < filterIds.length; i++) {
            assertEquals(filterIds[i], iterator.nextDoc());
            assertEquals(expectedScores.get(i), (Float) iterator.score());
        }
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }
}

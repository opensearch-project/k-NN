/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import lombok.SneakyThrows;
import org.apache.lucene.index.FieldInfo;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesIterator;

import java.util.HashMap;
import java.util.Map;

import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.KNNRestTestCase.FIELD_NAME;

public class RootPerFieldDerivedVectorInjectorTests extends KNNTestCase {
    public static float[] TEST_VECTOR = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };

    @SneakyThrows
    public void testInject() {
        FieldInfo fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder(FIELD_NAME).build();
        try (MockedStatic<KNNVectorValuesFactory> mockedKnnVectorValues = Mockito.mockStatic(KNNVectorValuesFactory.class)) {

            final KNNVectorValuesIterator vectorValuesIterator = Mockito.mock(KNNVectorValuesIterator.class);
            when(vectorValuesIterator.docId()).thenReturn(0);
            when(vectorValuesIterator.advance(anyInt())).thenReturn(0);
            when(vectorValuesIterator.nextDoc()).thenReturn(0);
            when(vectorValuesIterator.getDocIdSetIterator()).thenReturn(null);
            when(vectorValuesIterator.liveDocs()).thenReturn(0L);
            when(vectorValuesIterator.getVectorExtractorStrategy()).thenReturn(null);

            mockedKnnVectorValues.when(() -> KNNVectorValuesFactory.getVectorValues(fieldInfo, null, null))
                .thenReturn(new KNNVectorValues<float[]>(vectorValuesIterator) {
                    @Override
                    public float[] getVector() {
                        return TEST_VECTOR;
                    }

                    @Override
                    public float[] conditionalCloneVector() {
                        return TEST_VECTOR;
                    }
                });
            PerFieldDerivedVectorInjector perFieldDerivedVectorInjector = new RootPerFieldDerivedVectorInjector(
                fieldInfo,
                new DerivedSourceReaders(null, null, null, null, false)
            );

            Map<String, Object> source = new HashMap<>();
            perFieldDerivedVectorInjector.inject(0, source);
            assertArrayEquals(TEST_VECTOR, (float[]) source.get(FIELD_NAME), 0.0001f);
        }
    }
}

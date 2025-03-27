/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.backward_codecs.KNN9120Codec;

import lombok.SneakyThrows;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.search.DocIdSetIterator;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesIterator;
import org.opensearch.knn.index.vectorvalues.VectorValueExtractorStrategy;

import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.KNNRestTestCase.FIELD_NAME;

public class RootPerFieldDerivedVectorInjectorTests extends KNNTestCase {
    public static float[] TEST_VECTOR = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };

    @SneakyThrows
    public void testInject() {
        FieldInfo fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder(FIELD_NAME).build();
        try (MockedStatic<KNNVectorValuesFactory> mockedKnnVectorValues = Mockito.mockStatic(KNNVectorValuesFactory.class)) {
            mockedKnnVectorValues.when(() -> KNNVectorValuesFactory.getVectorValues(fieldInfo, null, null))
                .thenReturn(new KNNVectorValues<float[]>(new KNNVectorValuesIterator() {
                    @Override
                    public int docId() {
                        return 0;
                    }

                    @Override
                    public int advance(int docId) {
                        return 0;
                    }

                    @Override
                    public int nextDoc() {
                        return 0;
                    }

                    @Override
                    public DocIdSetIterator getDocIdSetIterator() {
                        return null;
                    }

                    @Override
                    public long liveDocs() {
                        return 0;
                    }

                    @Override
                    public VectorValueExtractorStrategy getVectorExtractorStrategy() {
                        return null;
                    }
                }) {

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
                new KNN9120DerivedSourceReaders(null, null, null, null)
            );

            Map<String, Object> source = new HashMap<>();
            perFieldDerivedVectorInjector.inject(0, source);
            assertArrayEquals(TEST_VECTOR, (float[]) source.get(FIELD_NAME), 0.0001f);
        }
    }
}

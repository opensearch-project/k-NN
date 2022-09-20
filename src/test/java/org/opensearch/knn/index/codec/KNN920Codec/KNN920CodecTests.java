/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN920Codec;

import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.KNNCodecTestCase;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ExecutionException;

import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.HNSW_ALGO_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.HNSW_ALGO_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.index.codec.KNNCodecFactory.CodecDelegateFactory.createKNN92DefaultDelegate;

public class KNN920CodecTests extends KNNCodecTestCase {

    public void testMultiFieldsKnnIndex() throws Exception {
        testMultiFieldsKnnIndex(KNN920Codec.builder().delegate(createKNN92DefaultDelegate()).build());
    }

    public void testBuildFromModelTemplate() throws InterruptedException, ExecutionException, IOException {
        testBuildFromModelTemplate((KNN920Codec.builder().delegate(createKNN92DefaultDelegate()).build()));
    }

    public void testKnnVectorIndex() throws Exception {
        final MapperService mapperService = mock(MapperService.class);
        final KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Map.of(HNSW_ALGO_M, 16, HNSW_ALGO_EF_CONSTRUCTION, 256))
        );
        final KNNVectorFieldMapper.KNNVectorFieldType mappedFieldType1 = new KNNVectorFieldMapper.KNNVectorFieldType(
            FIELD_NAME_ONE,
            Map.of(),
            3,
            knnMethodContext
        );
        final KNNVectorFieldMapper.KNNVectorFieldType mappedFieldType2 = new KNNVectorFieldMapper.KNNVectorFieldType(
            FIELD_NAME_TWO,
            Map.of(),
            2,
            knnMethodContext
        );
        when(mapperService.fieldType(eq(FIELD_NAME_ONE))).thenReturn(mappedFieldType1);
        when(mapperService.fieldType(eq(FIELD_NAME_TWO))).thenReturn(mappedFieldType2);

        var knnVectorsFormat = spy(new KNN920PerFieldKnnVectorsFormat(Optional.of(mapperService)));

        final KNN920Codec codec = KNN920Codec.builder().delegate(createKNN92DefaultDelegate()).knnVectorsFormat(knnVectorsFormat).build();

        testKnnVectorIndex(codec, knnVectorsFormat);
    }
}

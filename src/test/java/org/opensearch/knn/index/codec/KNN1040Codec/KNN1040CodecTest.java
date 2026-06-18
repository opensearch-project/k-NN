/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.backward_codecs.lucene99.Lucene99RWHnswScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.CustomCodec;
import org.opensearch.knn.index.codec.CustomCodecNoStoredFields;
import org.opensearch.knn.index.codec.KNNCodecTestCase;
import org.opensearch.knn.index.codec.KNNCodecVersion;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;

import static org.hamcrest.Matchers.instanceOf;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.index.engine.KNNEngine.LUCENE;

public class KNN1040CodecTest extends KNNCodecTestCase {

    @SneakyThrows
    public void testMultiFieldsKnnIndex() {
        testMultiFieldsKnnIndex(KNN1040Codec.builder().delegate(KNNCodecVersion.CURRENT_DEFAULT_DELEGATE).build());
    }

    @SneakyThrows
    public void testMultiFieldsKnnIndexCustomCodecWithStoredFields() {
        testMultiFieldsKnnIndex(KNN1040Codec.builder().delegate(new CustomCodec()).build());
    }

    @SneakyThrows
    public void testMultiFieldsKnnIndexCustomCodecWithoutStoredFields() {
        testMultiFieldsKnnIndex(KNN1040Codec.builder().delegate(new CustomCodecNoStoredFields()).build());
    }

    @SneakyThrows
    public void testBuildFromModelTemplate() {
        testBuildFromModelTemplate(KNN1040Codec.builder().delegate(KNNCodecVersion.CURRENT_DEFAULT_DELEGATE).build());
    }

    // Ensure that the codec is able to return the correct per field knn vectors format for codec
    public void testCodecSetsCustomPerFieldKnnVectorsFormat() {
        final Codec codec = new KNN1040Codec();
        assertTrue(codec.knnVectorsFormat() instanceof KNN1040PerFieldKnnVectorsFormat);
    }

    public void testFlatFormatResolver_returnsKNN1040ScalarQuantizedVectorsFormat() {
        KNNMethodContext flatMethodContext = new KNNMethodContext(
            LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_FLAT, Collections.emptyMap())
        );
        assertThat(resolveFormat(flatMethodContext), instanceOf(KNN1040ScalarQuantizedVectorsFormat.class));
    }

    public void testSQOneBitFormatResolver_returnsKNN1040HnswScalarQuantizedVectorsFormat() {
        KNNMethodContext sqMethodContext = buildHnswMethodContext(Map.of(LUCENE_SQ_BITS, 1));
        assertThat(resolveFormat(sqMethodContext), instanceOf(KNN1040HnswScalarQuantizedVectorsFormat.class));
    }

    public void testSQDefaultBitsFormatResolver_returnsLuceneRWHnswSQFormat() {
        KNNMethodContext sqMethodContext = buildHnswMethodContext(Collections.emptyMap());
        assertThat(resolveFormat(sqMethodContext), instanceOf(Lucene99RWHnswScalarQuantizedVectorsFormat.class));
    }

    // IMPORTANT: When this Codec is moved to a backwards Codec, this test needs to be removed, because it attempts to
    // write with a read-only codec, which will fail
    @SneakyThrows
    public void testKnnVectorIndex() {
        Function<MapperService, PerFieldKnnVectorsFormat> perFieldKnnVectorsFormatProvider = (
            mapperService) -> new KNN1040PerFieldKnnVectorsFormat(Optional.of(mapperService));

        Function<PerFieldKnnVectorsFormat, Codec> knnCodecProvider = (knnVectorFormat) -> KNN1040Codec.builder()
            .delegate(KNNCodecVersion.CURRENT_DEFAULT_DELEGATE)
            .knnVectorsFormat(knnVectorFormat)
            .build();

        testKnnVectorIndex(knnCodecProvider, perFieldKnnVectorsFormatProvider);
    }

    public void testHnswWithoutEncoderFormatResolver_returnsLucene99HnswFormat() {
        Map<String, Object> params = new HashMap<>();
        params.put(METHOD_PARAMETER_M, 32);
        params.put(METHOD_PARAMETER_EF_CONSTRUCTION, 256);

        KNNMethodContext hnswMethodContext = new KNNMethodContext(LUCENE, SpaceType.L2, new MethodComponentContext(METHOD_HNSW, params));
        assertThat(resolveFormat(hnswMethodContext), instanceOf(Lucene99HnswVectorsFormat.class));
    }

    public void testFaissSQOneBitFormatResolver_returnsFaiss1040ScalarQuantizedFormat() {
        // Faiss uses SQ_BITS while Lucene uses LUCENE_SQ_BITS for the encoder parameter key
        MethodComponentContext encoderContext = new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 1));
        MethodComponentContext hnswContext = new MethodComponentContext("hnsw", Map.of(METHOD_ENCODER_PARAMETER, encoderContext));
        KNNMethodContext methodContext = mock(KNNMethodContext.class);
        when(methodContext.getKnnEngine()).thenReturn(org.opensearch.knn.index.engine.KNNEngine.FAISS);
        when(methodContext.getMethodComponentContext()).thenReturn(hnswContext);

        assertThat(resolveFormat(methodContext), instanceOf(Faiss1040ScalarQuantizedKnnVectorsFormat.class));
    }

    public void testLuceneSQOneBitFormatResolver_returnsKNN1040HnswSQFormat() {
        KNNMethodContext sqMethodContext = buildHnswMethodContext(Map.of(LUCENE_SQ_BITS, 1));
        assertThat(resolveFormat(sqMethodContext), instanceOf(KNN1040HnswScalarQuantizedVectorsFormat.class));
    }

    private KNNMethodContext buildHnswMethodContext(Map<String, Object> encoderParams) {
        MethodComponentContext encoderContext = new MethodComponentContext(ENCODER_SQ, encoderParams);

        Map<String, Object> params = new HashMap<>();
        params.put(METHOD_ENCODER_PARAMETER, encoderContext);
        params.put(METHOD_PARAMETER_M, 16);
        params.put(METHOD_PARAMETER_EF_CONSTRUCTION, 100);

        return new KNNMethodContext(LUCENE, SpaceType.L2, new MethodComponentContext(METHOD_HNSW, params));
    }

    private KnnVectorsFormat resolveFormat(KNNMethodContext methodContext) {
        MapperService mapperService = mock(MapperService.class);
        KNNVectorFieldType fieldType = new KNNVectorFieldType(
            "test_field",
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            getMappingConfigForMethodMapping(methodContext, 3)
        );
        when(mapperService.fieldType(eq("test_field"))).thenReturn(fieldType);

        KNN1040PerFieldKnnVectorsFormat format = new KNN1040PerFieldKnnVectorsFormat(Optional.of(mapperService));
        return format.getKnnVectorsFormatForField("test_field");
    }
}

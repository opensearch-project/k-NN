/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.CustomCodec;
import org.opensearch.knn.index.codec.CustomCodecNoStoredFields;
import org.opensearch.knn.index.codec.KNNCodecTestCase;
import org.opensearch.knn.index.codec.KNNCodecVersion;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;

import java.util.Collections;
import java.util.Optional;
import java.util.function.Function;

import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;
import static org.opensearch.knn.index.engine.BuiltinKNNEngine.LUCENE;

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

        MapperService mapperService = mock(MapperService.class);
        KNNVectorFieldType fieldType = new KNNVectorFieldType(
            "test_field",
            Collections.emptyMap(),
            org.opensearch.knn.index.VectorDataType.FLOAT,
            getMappingConfigForMethodMapping(flatMethodContext, 3)
        );
        when(mapperService.fieldType(eq("test_field"))).thenReturn(fieldType);

        KNN1040PerFieldKnnVectorsFormat format = new KNN1040PerFieldKnnVectorsFormat(Optional.of(mapperService));
        KnnVectorsFormat result = format.getKnnVectorsFormatForField("test_field");
        assertTrue(
            "Expected KNN1040ScalarQuantizedVectorsFormat but got " + result.getClass().getSimpleName(),
            result instanceof KNN1040ScalarQuantizedVectorsFormat
        );
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

}

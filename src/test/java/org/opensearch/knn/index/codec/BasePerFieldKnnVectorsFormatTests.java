/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.junit.Before;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;
import org.opensearch.knn.index.codec.params.KNNVectorsFormatParams;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.KNNMappingConfig;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;
import org.opensearch.knn.index.mapper.ModelFieldMapper;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelUtil;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Optional;
import java.util.function.Function;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.when;
import static org.mockito.MockitoAnnotations.openMocks;

public class BasePerFieldKnnVectorsFormatTests extends OpenSearchTestCase {

    private static final String FIELD = "field";
    private static final String MODEL_ID = "model_id";

    @Mock
    private MapperService mapperService;
    @Mock
    private Function<KNNVectorsFormatParams, KnnVectorsFormat> vectorsFormatSupplier;
    @Mock
    private KNNMappingConfig knnMappingConfig;
    @Mock
    private KNNMethodContext knnMethodContext;
    @Mock
    private MethodComponentContext methodComponentContext;

    private BasePerFieldKnnVectorsFormat basePerFieldKnnVectorsFormat;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        openMocks(this);
        basePerFieldKnnVectorsFormat = new BasePerFieldKnnVectorsFormat(
            Optional.of(mapperService),
            10,
            10,
            Lucene99HnswVectorsFormat::new,
            vectorsFormatSupplier
        ) {
        };
    }

    public void testGetKNNVectorsFormatForField() {
        MappedFieldType mappedFieldType = mock(MappedFieldType.class);
        when(mapperService.fieldType(FIELD)).thenReturn(mappedFieldType);

        KnnVectorsFormat knnVectorsFormat = basePerFieldKnnVectorsFormat.getKnnVectorsFormatForField(FIELD);
        assertEquals(Lucene99HnswVectorsFormat.class, knnVectorsFormat.getClass());

        KNNVectorFieldType knnVectorFieldType = mock(KNNVectorFieldType.class);
        when(knnVectorFieldType.getKnnMappingConfig()).thenReturn(knnMappingConfig);
        when(knnMappingConfig.getKnnMethodContext()).thenReturn(Optional.of(knnMethodContext));
        when(knnMethodContext.getMethodComponentContext()).thenReturn(methodComponentContext);
        when(mapperService.fieldType(FIELD)).thenReturn(knnVectorFieldType);

        KnnVectorsFormat expected = new Lucene99HnswVectorsFormat(10, 10);
        when(vectorsFormatSupplier.apply(new KNNVectorsFormatParams(null, 10, 10))).thenReturn(expected);
        assertEquals(NativeEngines990KnnVectorsFormat.class, basePerFieldKnnVectorsFormat.getKnnVectorsFormatForField(FIELD).getClass());
    }

    public void testGetKNNVectorsFormatForFieldWithModel() {
        ModelMetadata metadata = mock(ModelMetadata.class);
        try (
            MockedStatic<ModelUtil> modelUtilMock = mockStatic(ModelUtil.class);
            MockedStatic<ModelFieldMapper> modelFieldMapperMock = mockStatic(ModelFieldMapper.class)
        ) {
            KNNVectorFieldType knnVectorFieldType = mock(KNNVectorFieldType.class);
            when(knnVectorFieldType.getKnnMappingConfig()).thenReturn(knnMappingConfig);
            when(knnMappingConfig.getModelId()).thenReturn(Optional.of(MODEL_ID));
            modelUtilMock.when(() -> ModelUtil.getModelMetadata(MODEL_ID)).thenReturn(metadata);
            modelFieldMapperMock.when(() -> ModelFieldMapper.getKNNMethodContextFromModelMetadata(metadata)).thenReturn(knnMethodContext);
            when(knnMethodContext.getMethodComponentContext()).thenReturn(methodComponentContext);
            when(mapperService.fieldType(FIELD)).thenReturn(knnVectorFieldType);

            assertEquals(
                NativeEngines990KnnVectorsFormat.class,
                basePerFieldKnnVectorsFormat.getKnnVectorsFormatForField(FIELD).getClass()
            );
        }
    }
}

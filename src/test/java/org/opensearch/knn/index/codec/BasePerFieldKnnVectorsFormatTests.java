/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import org.opensearch.knn.KNNTestCase;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.mockito.MockedConstruction;
import org.mockito.Mockito;
import org.opensearch.Version;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.KNNMappingConfig;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.Collections;
import java.util.Optional;

import org.opensearch.knn.index.codec.params.KNNBBQVectorsFormatParams;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

public class BasePerFieldKnnVectorsFormatTests extends KNNTestCase {

    private static MapperService mockMapperService;
    private static KNNVectorFieldType mockFieldType;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        mockMapperService = Mockito.mock(MapperService.class);
        mockFieldType = Mockito.mock(KNNVectorFieldType.class);
        Mockito.when(mockMapperService.fieldType("field")).thenReturn(mockFieldType);
        IndexSettings mockIndexSettings = Mockito.mock(IndexSettings.class);
        Mockito.when(mockIndexSettings.getValue(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_SETTING))
            .thenReturn(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE);
        Mockito.when(mockMapperService.getIndexSettings()).thenReturn(mockIndexSettings);
    }

    private static class StubKnnVectorsFormat extends KnnVectorsFormat {
        StubKnnVectorsFormat() {
            super("stub");
        }

        @Override
        public KnnVectorsWriter fieldsWriter(SegmentWriteState s) {
            return null;
        }

        @Override
        public KnnVectorsReader fieldsReader(SegmentReadState s) {
            return null;
        }

        @Override
        public int getMaxDimensions(String fieldName) {
            return 16000;
        }
    }

    private static class TestPerFieldKnnVectorsFormat extends BasePerFieldKnnVectorsFormat {
        static final KnnVectorsFormat DEFAULT_FORMAT = new StubKnnVectorsFormat();
        static final KnnVectorsFormat VECTORS_FORMAT = new StubKnnVectorsFormat();
        static final KnnVectorsFormat SCALAR_FORMAT = new StubKnnVectorsFormat();
        static final KnnVectorsFormat BBQ_FORMAT = new StubKnnVectorsFormat();

        TestPerFieldKnnVectorsFormat(Optional<MapperService> mapperService) {
            super(
                mapperService,
                16,
                256,
                () -> DEFAULT_FORMAT,
                p -> VECTORS_FORMAT,
                p -> SCALAR_FORMAT,
                p -> BBQ_FORMAT,
                Mockito.mock(NativeIndexBuildStrategyFactory.class)
            );
        }

        @Override
        public int getMaxDimensions(String fieldName) {
            return 16000;
        }
    }

    private static KNNMethodContext getKnnMethodContext(KNNEngine engine) {
        return new KNNMethodContext(engine, SpaceType.L2, new MethodComponentContext(METHOD_HNSW, Collections.emptyMap()));
    }

    private static KNNMappingConfig getMappingConfig(
        CompressionLevel compressionLevel,
        KNNMethodContext knnMethodContext,
        Mode mode,
        Version indexCreatedVersion,
        Optional<String> topLevelEngine
    ) {
        return new KNNMappingConfig() {
            @Override
            public Optional<KNNMethodContext> getKnnMethodContext() {
                return Optional.of(knnMethodContext);
            }

            @Override
            public int getDimension() {
                return 4;
            }

            @Override
            public Mode getMode() {
                return mode;
            }

            @Override
            public Version getIndexCreatedVersion() {
                return indexCreatedVersion;
            }

            @Override
            public CompressionLevel getCompressionLevel() {
                return compressionLevel;
            }
        };
    }

    public void testBasePerFieldKnnVectorsFormat_whenVersion36LuceneOnDisk_thenReturnBBQFormat() {
        KNNMethodContext knnMethodContext = getKnnMethodContext(KNNEngine.LUCENE);
        KNNMappingConfig mappingConfig = getMappingConfig(
            CompressionLevel.NOT_CONFIGURED,
            knnMethodContext,
            Mode.ON_DISK,
            Version.V_3_6_0,
            Optional.empty()
        );
        Mockito.when(mockFieldType.getKnnMappingConfig()).thenReturn(mappingConfig);

        try (MockedConstruction<KNNBBQVectorsFormatParams> ignored = Mockito.mockConstruction(KNNBBQVectorsFormatParams.class)) {
            TestPerFieldKnnVectorsFormat format = new TestPerFieldKnnVectorsFormat(Optional.of(mockMapperService));
            assertSame(TestPerFieldKnnVectorsFormat.BBQ_FORMAT, format.getKnnVectorsFormatForField("field"));
        }
    }

    public void testBasePerFieldKnnVectorsFormat_whenVersion36EngineNotLuceneOnDisk_thenDontReturnBBQFormat() {
        KNNMethodContext knnMethodContext = getKnnMethodContext(KNNEngine.FAISS);
        KNNMappingConfig mappingConfig = getMappingConfig(
            CompressionLevel.NOT_CONFIGURED,
            knnMethodContext,
            Mode.ON_DISK,
            Version.V_3_6_0,
            Optional.empty()
        );
        Mockito.when(mockFieldType.getKnnMappingConfig()).thenReturn(mappingConfig);

        TestPerFieldKnnVectorsFormat format = new TestPerFieldKnnVectorsFormat(Optional.of(mockMapperService));
        assertNotSame(TestPerFieldKnnVectorsFormat.BBQ_FORMAT, format.getKnnVectorsFormatForField("field"));
    }

    public void testBasePerFieldKnnVectorsFormat_whenVersion36LuceneCompression32x_thenReturnBBQFormat() {
        KNNMethodContext knnMethodContext = getKnnMethodContext(KNNEngine.LUCENE);
        KNNMappingConfig mappingConfig = getMappingConfig(
            CompressionLevel.x32,
            knnMethodContext,
            Mode.IN_MEMORY,
            Version.V_3_6_0,
            Optional.empty()
        );
        Mockito.when(mockFieldType.getKnnMappingConfig()).thenReturn(mappingConfig);

        try (MockedConstruction<KNNBBQVectorsFormatParams> ignored = Mockito.mockConstruction(KNNBBQVectorsFormatParams.class)) {
            TestPerFieldKnnVectorsFormat format = new TestPerFieldKnnVectorsFormat(Optional.of(mockMapperService));
            assertSame(TestPerFieldKnnVectorsFormat.BBQ_FORMAT, format.getKnnVectorsFormatForField("field"));
        }
    }

    public void testBasePerFieldKnnVectorsFormat_whenVersionNot36LuceneOnDisk_thenDontReturnBBQFormat() {
        KNNMethodContext knnMethodContext = getKnnMethodContext(KNNEngine.LUCENE);
        KNNMappingConfig mappingConfig = getMappingConfig(
            CompressionLevel.NOT_CONFIGURED,
            knnMethodContext,
            Mode.ON_DISK,
            Version.V_3_3_0,
            Optional.empty()
        );
        Mockito.when(mockFieldType.getKnnMappingConfig()).thenReturn(mappingConfig);

        TestPerFieldKnnVectorsFormat format = new TestPerFieldKnnVectorsFormat(Optional.of(mockMapperService));
        assertNotSame(TestPerFieldKnnVectorsFormat.BBQ_FORMAT, format.getKnnVectorsFormatForField("field"));
    }

    public void testBasePerFieldKnnVectorsFormat_whenVersion36LuceneOnDiskNot32xCompression_thenDontReturnBBQFormat() {
        KNNMethodContext knnMethodContext = getKnnMethodContext(KNNEngine.LUCENE);
        KNNMappingConfig mappingConfig = getMappingConfig(
            CompressionLevel.x16,
            knnMethodContext,
            Mode.ON_DISK,
            Version.V_3_6_0,
            Optional.empty()
        );
        Mockito.when(mockFieldType.getKnnMappingConfig()).thenReturn(mappingConfig);

        TestPerFieldKnnVectorsFormat format = new TestPerFieldKnnVectorsFormat(Optional.of(mockMapperService));
        assertNotSame(TestPerFieldKnnVectorsFormat.BBQ_FORMAT, format.getKnnVectorsFormatForField("field"));
    }
}

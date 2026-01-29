/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import org.apache.logging.log4j.Logger;
import org.apache.lucene.codecs.Codec;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.index.Index;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.codec.CodecServiceConfig;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;

import java.util.List;
import java.util.UUID;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Test for KNNCodecService class with focus on codec by name lookup
 */
public class KNNCodecServiceTests extends KNNTestCase {
    private static final String TEST_INDEX = "test-index";
    private static final int NUM_OF_SHARDS = 1;
    private static final UUID INDEX_UUID = UUID.randomUUID();

    private IndexSettings indexSettings;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        IndexMetadata indexMetadata = mock(IndexMetadata.class);
        when(indexMetadata.getIndex()).thenReturn(new Index(TEST_INDEX, INDEX_UUID.toString()));
        when(indexMetadata.getCustomData(IndexMetadata.REMOTE_STORE_CUSTOM_KEY)).thenReturn(null);
        when(indexMetadata.getSettings()).thenReturn(Settings.EMPTY);
        Settings settings = Settings.builder().put(IndexMetadata.SETTING_NUMBER_OF_SHARDS, Integer.toString(NUM_OF_SHARDS)).build();
        indexSettings = new IndexSettings(indexMetadata, settings);
    }

    public void testGetCodecByName() {
        MapperService mapperService = mock(MapperService.class);
        Logger loggerMock = mock(Logger.class);
        CodecServiceConfig codecServiceConfig = new CodecServiceConfig(indexSettings, mapperService, loggerMock, List.of());
        KNNCodecService knnCodecService = new KNNCodecService(codecServiceConfig, mock(NativeIndexBuildStrategyFactory.class));
        Codec codec = knnCodecService.codec(KNNCodecVersion.CURRENT_DEFAULT_DELEGATE.getName());
        assertNotNull(codec);
    }

    /**
     * This test case covers scenarios when MapperService is null, for instance this may happen during index.close operation.
     * In such scenario codec is not really required but is created as part of engine initialization, please check code references below:
     * @see <a href="https://github.com/opensearch-project/OpenSearch/blob/main/server/src/main/java/org/opensearch/index/engine/EngineConfig.java#L378">EngineConfig.java</a>
     * @see <a href="https://github.com/opensearch-project/OpenSearch/blob/main/server/src/main/java/org/opensearch/index/shard/IndexShard.java#L3315">IndexShard.java</a>
     * @see <a href="https://github.com/opensearch-project/OpenSearch/blob/main/server/src/main/java/org/opensearch/index/engine/Engine.java#L906">Engine.java</a>
     */
    public void testGetCodecByNameWithNoMapperService() {
        Logger loggerMock = mock(Logger.class);
        CodecServiceConfig codecServiceConfig = new CodecServiceConfig(indexSettings, null, loggerMock, List.of());
        KNNCodecService knnCodecService = new KNNCodecService(codecServiceConfig, mock(NativeIndexBuildStrategyFactory.class));
        Codec codec = knnCodecService.codec(KNNCodecVersion.CURRENT_DEFAULT_DELEGATE.getName());
        assertNotNull(codec);
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import org.opensearch.common.settings.Setting;
import org.opensearch.index.codec.CodecServiceConfig;
import org.apache.lucene.codecs.Codec;
import org.opensearch.index.codec.CodecService;
import org.opensearch.index.mapper.MapperService;

import static org.opensearch.common.settings.Setting.Property.Dynamic;
import static org.opensearch.common.settings.Setting.Property.IndexScope;
import static org.opensearch.common.settings.Setting.Property.NodeScope;

/**
 * KNNCodecService to inject the right KNNCodec version
 */
public class KNNCodecService extends CodecService {
    // Setting to determine if index should use custom codec
    public static final String KNN_INDEX = "index.knn";
    public static final Setting<Boolean> IS_KNN_INDEX_SETTING = Setting.boolSetting(KNN_INDEX, false, IndexScope);

    // JNI Related settings
    public static final String KNN_ALGO_PARAM_INDEX_THREAD_QTY = "knn.algo_param.index_thread_qty";
    public static final Integer KNN_DEFAULT_ALGO_PARAM_INDEX_THREAD_QTY = 1;
    private static final int KNN_MAX_ALGO_PARAM_INDEX_THREAD_QTY = 32;

    /**
     * index_thread_quantity - the parameter specifies how many threads the nms library should use to create the graph.
     * By default, the nms library sets this value to NUM_CORES. However, because ES can spawn NUM_CORES threads for
     * indexing, and each indexing thread calls the NMS library to build the graph, which can also spawn NUM_CORES threads,
     * this could lead to NUM_CORES^2 threads running and could lead to 100% CPU utilization. This setting allows users to
     * configure number of threads for graph construction.
     */
    public static final Setting<Integer> KNN_ALGO_PARAM_INDEX_THREAD_QTY_SETTING = Setting.intSetting(
        KNN_ALGO_PARAM_INDEX_THREAD_QTY,
        KNN_DEFAULT_ALGO_PARAM_INDEX_THREAD_QTY,
        1,
        KNN_MAX_ALGO_PARAM_INDEX_THREAD_QTY,
        NodeScope,
        Dynamic
    );

    private final MapperService mapperService;

    public KNNCodecService(CodecServiceConfig codecServiceConfig) {
        super(codecServiceConfig.getMapperService(), codecServiceConfig.getIndexSettings(), codecServiceConfig.getLogger());
        mapperService = codecServiceConfig.getMapperService();
    }

    /**
     * Return the custom k-NN codec that wraps another codec that a user wants to use for non k-NN related operations
     *
     * @param name of delegate codec.
     * @return Latest KNN Codec built with delegate codec.
     */
    @Override
    public Codec codec(String name) {
        return KNNCodecVersion.current().getKnnCodecSupplier().apply(super.codec(name), mapperService);
    }
}

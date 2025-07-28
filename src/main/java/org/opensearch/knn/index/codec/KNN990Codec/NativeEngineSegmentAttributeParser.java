/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import org.apache.commons.lang.StringUtils;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.engine.MemoryOptimizedSearchSupportSpec;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;

import java.util.Arrays;
import java.util.Collections;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

public class NativeEngineSegmentAttributeParser {
    static final String WARMUP_ENABLED = "warmup_enabled";
    static final String INDEX_NAME = "index_name";
    static final String TRUE = "true";
    static final String MEMORY_OPTIMIZED_FIELDS = "memory_optimized";
    static final String DELIMITER = ",";

    /**
     * From segmentInfo, parse whether warmup is enabled
     *
     * @param segmentInfo {@link SegmentInfo}
     * @return boolean of whether the warmup setting is enabled
     */
    public static boolean parseWarmup(SegmentInfo segmentInfo) {
        if (segmentInfo == null) {
            throw new IllegalArgumentException("SegmentInfo cannot be null");
        }
        return segmentInfo.getAttribute(WARMUP_ENABLED) != null && segmentInfo.getAttribute(INDEX_NAME) != null;
    }

    /**
     * From segmentInfo, parse index name
     *
     * @param segmentInfo {@link SegmentInfo}
     * @return String of the name of the index the segment is a member of
     */
    public static String parseIndexName(SegmentInfo segmentInfo) {
        if (segmentInfo == null) {
            throw new IllegalArgumentException("SegmentInfo cannot be null");
        }
        return segmentInfo.getAttribute(INDEX_NAME);
    }

    public static Set<String> parseMemoryOptimizedFields(SegmentInfo segmentInfo) {
        if (segmentInfo == null) {
            throw new IllegalArgumentException("SegmentInfo cannot be null");
        }
        String memoryOptimizedFields = segmentInfo.getAttribute(MEMORY_OPTIMIZED_FIELDS);
        if (StringUtils.isEmpty(memoryOptimizedFields)) {
            return Collections.emptySet();
        }
        return Arrays.stream(memoryOptimizedFields.split(DELIMITER, 0)).collect(Collectors.toSet());
    }

    /**
     * Adds {@link SegmentInfo} attribute for warmup
     *
     * @param segmentWriteState {@link SegmentWriteState}
     */
    public static void addWarmupSegmentInfoAttribute(MapperService mapperService, SegmentWriteState segmentWriteState) {
        if (segmentWriteState == null) {
            throw new IllegalArgumentException("SegmentWriteState cannot be null");
        }
        SegmentInfo segmentInfo = segmentWriteState.segmentInfo;
        String indexName = mapperService.index().getName();
        segmentInfo.putAttribute(INDEX_NAME, indexName);
        segmentInfo.putAttribute(WARMUP_ENABLED, TRUE);
        if (segmentWriteState.fieldInfos != null) {
            final Set<String> fieldsForMemoryOptimizedSearch = StreamSupport.stream(segmentWriteState.fieldInfos.spliterator(), false)
                .filter(fieldInfo -> fieldInfo.attributes().containsKey(KNNVectorFieldMapper.KNN_FIELD))
                .map(FieldInfo::getName)
                .filter(name -> {
                    final MappedFieldType fieldType = mapperService.fieldType(name);
                    if (fieldType instanceof KNNVectorFieldType knnFieldType) {
                        return MemoryOptimizedSearchSupportSpec.isSupportedFieldType(knnFieldType, indexName);
                    }
                    return false;
                })
                .collect(Collectors.toSet());
            segmentInfo.putAttribute(MEMORY_OPTIMIZED_FIELDS, String.join(DELIMITER, fieldsForMemoryOptimizedSearch));
        }
    }
}

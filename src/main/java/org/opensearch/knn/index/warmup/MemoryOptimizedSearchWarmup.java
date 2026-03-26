/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.warmup;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.SegmentReader;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Log4j2
public class MemoryOptimizedSearchWarmup {
    public List<String> warmUp(final LeafReader leafReader, final MapperService mapperService, final String indexName) {
        if (mapperService == null) {
            return Collections.emptyList();
        }

        final SegmentReader segmentReader = Lucene.segmentReader(leafReader);

        final List<FieldInfo> memOptSearchFields = getFieldsForMemoryOptimizedSearch(leafReader, mapperService);
        final List<String> warmedUp = new ArrayList<>();

        for (FieldInfo field : memOptSearchFields) {
            if (warmUpField(field, segmentReader)) {
                warmedUp.add(field.getName());
            }
        }

        return warmedUp;
    }

    private boolean warmUpField(final FieldInfo field, final SegmentReader segmentReader) {
        try {
            segmentReader.getVectorReader().search(field.getName(), (float[]) null, WarmUpCollector.INSTANCE, null); // codecov[ignore]
            return true;
        } catch (Exception e) {
            // Expected during warmup initialization
            log.warn("Warm up failed for {}", field.getName(), e);
            return false;
        }
    }

    private boolean isMemoryOptimizedSearchField(final FieldInfo fieldInfo, final MapperService mapperService) {
        if (fieldInfo.attributes().containsKey(KNNVectorFieldMapper.KNN_FIELD) == false) {
            return false;
        }
        final MappedFieldType fieldType = mapperService.fieldType(fieldInfo.getName());
        return fieldType instanceof KNNVectorFieldType knnFieldType && knnFieldType.isMemoryOptimizedSearchAvailable();
    }

    private List<FieldInfo> getFieldsForMemoryOptimizedSearch(final LeafReader leafReader, final MapperService mapperService) {
        final List<FieldInfo> fields = new ArrayList<>();
        for (FieldInfo field : leafReader.getFieldInfos()) {
            if (isMemoryOptimizedSearchField(field, mapperService)) {
                fields.add(field);
            }
        }
        return fields;
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.warmup;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.engine.MemoryOptimizedSearchSupportSpec;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;

import java.io.IOException;
import java.util.ArrayList;

@Log4j2
public class MemoryOptimizedSearchWarmup {

    public ArrayList<String> warmUp(
        final LeafReader leafReader,
        final MapperService mapperService,
        final String indexName,
        final Directory directory
    ) {
        if (mapperService == null) {
            return new ArrayList<>();
        }
        FieldWarmUpStrategy fieldWarmUpStrategy = new FieldWarmUpStrategyFactory().setLeafReader(leafReader)
            .setDirectory(directory)
            .build();

        ArrayList<FieldInfo> memOptSearchFields = getFieldsForMemoryOptimizedSearch(leafReader, mapperService, indexName);
        for (FieldInfo field : memOptSearchFields) {
            loadFullPrecisionVectors(leafReader, field);
        }

        ArrayList<String> warmedUp = new ArrayList<>();

        for (FieldInfo field : memOptSearchFields) {
            try {
                if (fieldWarmUpStrategy.warmUp(field)) {
                    warmedUp.add(field.getName());
                }
            } catch (IOException e) {
                log.error("Failed to warm up field: {}", field.getName());
            }
        }

        return warmedUp;
    }

    private boolean isMemoryOptimizedSearchField(FieldInfo fieldInfo, MapperService mapperService, String indexName) {
        if (fieldInfo.attributes().containsKey(KNNVectorFieldMapper.KNN_FIELD) == false) {
            return false;
        }

        final MappedFieldType fieldType = mapperService.fieldType(fieldInfo.getName());

        if (fieldType instanceof KNNVectorFieldType knnFieldType) {
            return MemoryOptimizedSearchSupportSpec.isSupportedFieldType(knnFieldType, indexName);
        }

        return false;
    }

    private ArrayList<FieldInfo> getFieldsForMemoryOptimizedSearch(LeafReader leafReader, MapperService mapperService, String indexName) {
        ArrayList<FieldInfo> fields = new ArrayList<>();
        for (FieldInfo field : leafReader.getFieldInfos()) {
            if (isMemoryOptimizedSearchField(field, mapperService, indexName)) {
                fields.add(field);
            }
        }
        return fields;
    }

    private void loadFullPrecisionVectors(LeafReader leafReader, FieldInfo field) {
        try {
            final FloatVectorValues vectorValues = leafReader.getFloatVectorValues(field.getName());
            if (vectorValues == null) {
                return;
            }

            final KnnVectorValues.DocIndexIterator iter = vectorValues.iterator();
            while (iter.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                vectorValues.vectorValue(iter.docID());
            }
        } catch (IOException e) {
            log.error("Failed to load vec file for field: {}", field.getName(), e);
        }
    }
}

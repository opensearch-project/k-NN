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
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.MemoryOptimizedSearchSupportSpec;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import static org.opensearch.knn.common.FieldInfoExtractor.extractKNNEngine;

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

        final SegmentReader segmentReader = Lucene.segmentReader(leafReader);
        final Directory bottomDirectory = FilterDirectory.unwrap(directory);

        ArrayList<FieldInfo> memOptSearchFields = getFieldsForMemoryOptimizedSearch(leafReader, mapperService, indexName);
        for (FieldInfo field : memOptSearchFields) {
            loadFullPrecisionVectors(leafReader, field);
        }

        ArrayList<String> warmedUp = new ArrayList<>();

        for (FieldInfo field : memOptSearchFields) {
            try {
                if (warmUpField(field, segmentReader, bottomDirectory)) {
                    warmedUp.add(field.getName());
                }
            } catch (IOException e) {
                log.error("Failed to warm up field: {}", field.getName(), e);
            }
        }

        return warmedUp;
    }

    private boolean warmUpField(FieldInfo field, SegmentReader segmentReader, Directory directory) throws IOException {
        final KNNEngine knnEngine = extractKNNEngine(field);
        final List<String> engineFiles = KNNCodecUtil.getEngineFiles(
            knnEngine.getExtension(),
            field.getName(),
            segmentReader.getSegmentInfo().info
        );
        if (engineFiles.isEmpty()) {
            log.warn("Could not find an engine file for field [{}]", field.getName());
            return false;
        }
        final Path indexPath = Paths.get(engineFiles.getFirst());

        try (IndexInput input = directory.openInput(indexPath.toString(), IOContext.READONCE)) {
            if (input.length() != 0) {
                for (long i = 0; i < input.length(); i += 4096) {
                    input.seek(i);
                    input.readByte();
                }
                input.seek(input.length() - 1);
                input.readByte();
            }
        }

        return true;
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

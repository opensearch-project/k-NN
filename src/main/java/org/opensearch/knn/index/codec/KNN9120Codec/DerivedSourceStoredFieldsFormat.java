/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

import lombok.AllArgsConstructor;
import org.apache.lucene.codecs.StoredFieldsFormat;
import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.codecs.StoredFieldsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.opensearch.common.Nullable;
import org.opensearch.index.mapper.FieldMapper;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.derivedsource.DerivedSourceReadersSupplier;
import org.opensearch.knn.index.codec.derivedsource.DerivedSourceSegmentAttributeHelper;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@AllArgsConstructor
public class DerivedSourceStoredFieldsFormat extends StoredFieldsFormat {

    private final StoredFieldsFormat delegate;
    private final DerivedSourceReadersSupplier derivedSourceReadersSupplier;
    // IMPORTANT Do not rely on this for the reader, it will be null if SPI is used
    @Nullable
    private final MapperService mapperService;

    @Override
    public StoredFieldsReader fieldsReader(Directory directory, SegmentInfo segmentInfo, FieldInfos fieldInfos, IOContext ioContext)
        throws IOException {
        List<FieldInfo> derivedVectorFields = DerivedSourceSegmentAttributeHelper.parseDerivedVectorFields(segmentInfo, fieldInfos);
        if (derivedVectorFields == null || derivedVectorFields.isEmpty()) {
            return delegate.fieldsReader(directory, segmentInfo, fieldInfos, ioContext);
        }
        Map<String, List<String>> nestedLineageMap = DerivedSourceSegmentAttributeHelper.parseNestedLineageMap(
            derivedVectorFields,
            segmentInfo
        );
        return new DerivedSourceStoredFieldsReader(
            delegate.fieldsReader(directory, segmentInfo, fieldInfos, ioContext),
            derivedVectorFields,
            nestedLineageMap,
            derivedSourceReadersSupplier,
            new SegmentReadState(directory, segmentInfo, fieldInfos, ioContext)
        );
    }

    @Override
    public StoredFieldsWriter fieldsWriter(Directory directory, SegmentInfo segmentInfo, IOContext ioContext) throws IOException {
        StoredFieldsWriter delegateWriter = delegate.fieldsWriter(directory, segmentInfo, ioContext);
        if (mapperService == null) {
            return delegateWriter;
        }

        // If there are excluded fields, we will not derive any vector fields
        if (mapperService.documentMapper().sourceMapper().enabled() == false
            || mapperService.documentMapper().sourceMapper().isComplete() == false) {
            return delegateWriter;
        }

        // Skipping composite indices for now
        if (mapperService.isCompositeIndexPresent() == true) {
            return delegateWriter;
        }

        // Check if the feature is enabled.
        if (KNNSettings.isKNNDerivedSourceEnabled(mapperService.getIndexSettings().getSettings()) == false) {
            return delegateWriter;
        }

        List<String> vectorFieldTypes = new ArrayList<>();
        List<List<String>> nestedLineagesForAllFields = new ArrayList<>();
        for (MappedFieldType fieldType : mapperService.fieldTypes()) {
            if (fieldType instanceof KNNVectorFieldType == false) {
                continue;
            }

            KNNVectorFieldType vectorFieldType = (KNNVectorFieldType) fieldType;

            // Skip copy to fields
            if (mapperService.documentMapper().mappers().getMapper(fieldType.name()) instanceof FieldMapper) {
                FieldMapper mapper = ((FieldMapper) mapperService.documentMapper().mappers().getMapper(fieldType.name()));
                if (mapper.copyTo() != null
                    && mapper.copyTo().copyToFields() != null
                    && mapper.copyTo().copyToFields().isEmpty() == false) {
                    continue;
                }
            }

            List<String> nestedLineageForField = new ArrayList<>();
            for (String parentPath = mapperService.documentMapper()
                .mappers()
                .getNestedScope(vectorFieldType.name()); parentPath != null; parentPath = mapperService.documentMapper()
                    .mappers()
                    .getNestedScope(parentPath)) {
                nestedLineageForField.add(parentPath);
            }

            // Only support one level of nesting
            if (nestedLineageForField.size() > 1) {
                continue;
            }

            nestedLineagesForAllFields.add(nestedLineageForField);
            vectorFieldTypes.add(vectorFieldType.name());
        }

        if (vectorFieldTypes.isEmpty()) {
            return delegateWriter;
        }

        DerivedSourceSegmentAttributeHelper.addDervicedVectorFieldsSegmentInfoAttribue(segmentInfo, vectorFieldTypes);
        DerivedSourceSegmentAttributeHelper.addNestedLineageSegmentInfoAttribute(segmentInfo, nestedLineagesForAllFields);
        return new DerivedSourceStoredFieldsWriter(delegateWriter, vectorFieldTypes);
    }
}

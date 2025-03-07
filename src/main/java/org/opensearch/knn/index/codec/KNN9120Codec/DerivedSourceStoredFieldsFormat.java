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
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.derivedsource.DerivedSourceReadersSupplier;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

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
        List<FieldInfo> derivedVectorFields = null;
        if (segmentInfo.getAttribute("derived_vector_fields") == null) {
            return delegate.fieldsReader(directory, segmentInfo, fieldInfos, ioContext);
        }
        String[] vectorFieldTypes = segmentInfo.getAttribute("derived_vector_fields").split(",");
        if (vectorFieldTypes.length == 0) {
            return delegate.fieldsReader(directory, segmentInfo, fieldInfos, ioContext);
        }
        for (String vectorFieldType : vectorFieldTypes) {
            if (derivedVectorFields == null) {
                derivedVectorFields = new ArrayList<>();
            }
            FieldInfo fieldInfo = fieldInfos.fieldInfo(vectorFieldType);
            if (fieldInfo != null) {
                derivedVectorFields.add(fieldInfo);
            }

        }
        return new DerivedSourceStoredFieldsReader(
            delegate.fieldsReader(directory, segmentInfo, fieldInfos, ioContext),
            derivedVectorFields,
            derivedSourceReadersSupplier,
            new SegmentReadState(directory, segmentInfo, fieldInfos, ioContext)
        );
    }

    @Override
    public StoredFieldsWriter fieldsWriter(Directory directory, SegmentInfo segmentInfo, IOContext ioContext) throws IOException {
        StoredFieldsWriter delegateWriter = delegate.fieldsWriter(directory, segmentInfo, ioContext);
        if (mapperService != null && KNNSettings.isKNNDerivedSourceEnabled(mapperService.getIndexSettings().getSettings())) {
            List<String> vectorFieldTypes = new ArrayList<>();
            for (MappedFieldType fieldType : mapperService.fieldTypes()) {
                if (fieldType instanceof KNNVectorFieldType
                    && KNNVectorFieldMapperUtil.isDeriveSourceForFieldEnabled(true, fieldType.name())) {
                    vectorFieldTypes.add(fieldType.name());
                }
            }
            if (vectorFieldTypes.isEmpty() == false) {
                segmentInfo.putAttribute("derived_vector_fields", String.join(",", vectorFieldTypes));
                return new DerivedSourceStoredFieldsWriter(delegateWriter, vectorFieldTypes);
            }
        }
        return delegateWriter;
    }
}

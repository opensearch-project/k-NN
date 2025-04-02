/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.backward_codecs.KNN9120Codec;

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
import org.opensearch.index.mapper.MapperService;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@AllArgsConstructor
public class KNN9120DerivedSourceStoredFieldsFormat extends StoredFieldsFormat {

    private final StoredFieldsFormat delegate;
    private final KNN9120DerivedSourceReadersSupplier derivedSourceReadersSupplier;
    // IMPORTANT Do not rely on this for the reader, it will be null if SPI is used
    @Nullable
    private final MapperService mapperService;

    static final String DERIVED_VECTOR_FIELD_ATTRIBUTE_KEY = "knn-derived-source-enabled";
    static final String DERIVED_VECTOR_FIELD_ATTRIBUTE_TRUE_VALUE = "true";

    @Override
    public StoredFieldsReader fieldsReader(Directory directory, SegmentInfo segmentInfo, FieldInfos fieldInfos, IOContext ioContext)
        throws IOException {
        List<FieldInfo> derivedVectorFields = null;
        for (FieldInfo fieldInfo : fieldInfos) {
            if (DERIVED_VECTOR_FIELD_ATTRIBUTE_TRUE_VALUE.equals(fieldInfo.attributes().get(DERIVED_VECTOR_FIELD_ATTRIBUTE_KEY))) {
                // Lazily initialize the list of fields
                if (derivedVectorFields == null) {
                    derivedVectorFields = new ArrayList<>();
                }
                derivedVectorFields.add(fieldInfo);
            }
        }
        // If no fields have it enabled, we can just short-circuit and return the delegate's fieldReader
        if (derivedVectorFields == null || derivedVectorFields.isEmpty()) {
            return delegate.fieldsReader(directory, segmentInfo, fieldInfos, ioContext);
        }
        return new KNN9120DerivedSourceStoredFieldsReader(
            delegate.fieldsReader(directory, segmentInfo, fieldInfos, ioContext),
            derivedVectorFields,
            derivedSourceReadersSupplier,
            new SegmentReadState(directory, segmentInfo, fieldInfos, ioContext)
        );
    }

    @Override
    public StoredFieldsWriter fieldsWriter(Directory directory, SegmentInfo segmentInfo, IOContext ioContext) {
        throw new UnsupportedOperationException("KNN9120DerivedSourceStoredFieldsFormat does not support fieldsWriter");
    }
}

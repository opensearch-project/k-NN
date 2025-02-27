/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import lombok.AllArgsConstructor;
import org.apache.lucene.codecs.Codec;
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
import org.opensearch.knn.index.mapper.KNNVectorFieldType;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.opensearch.knn.common.KNNConstants.DERIVED_VECTOR_FIELD_ATTRIBUTE_KEY;
import static org.opensearch.knn.common.KNNConstants.DERIVED_VECTOR_FIELD_ATTRIBUTE_TRUE_VALUE;

@AllArgsConstructor
public class DerivedSourceStoredFieldsFormat extends StoredFieldsFormat {

    private static final String DELEGATE_CODEC_KEY = "knn_delegate_codec";

    private final StoredFieldsFormat delegate;
    private final DerivedSourceReadersSupplier derivedSourceReadersSupplier;
    // IMPORTANT Do not rely on this for the reader, it will be null if SPI is used
    @Nullable
    private final MapperService mapperService;
    // IMPORTANT Do not rely on this for the reader, it will be null if SPI is used
    @Nullable
    private final String delegateCodecName;

    @Override
    public StoredFieldsReader fieldsReader(Directory directory, SegmentInfo segmentInfo, FieldInfos fieldInfos, IOContext ioContext)
        throws IOException {
        StoredFieldsFormat delegateFromWriting = delegate;
        if (segmentInfo.getAttribute(DELEGATE_CODEC_KEY) != null) {
            String delegateCodecName = segmentInfo.getAttribute(DELEGATE_CODEC_KEY);
            delegateFromWriting = Codec.forName(delegateCodecName).storedFieldsFormat();
        }

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
            return delegateFromWriting.fieldsReader(directory, segmentInfo, fieldInfos, ioContext);
        }
        return new DerivedSourceStoredFieldsReader(
            delegateFromWriting.fieldsReader(directory, segmentInfo, fieldInfos, ioContext),
            derivedVectorFields,
            derivedSourceReadersSupplier,
            new SegmentReadState(directory, segmentInfo, fieldInfos, ioContext)
        );
    }

    @Override
    public StoredFieldsWriter fieldsWriter(Directory directory, SegmentInfo segmentInfo, IOContext ioContext) throws IOException {
        // We write the delegate codec name into the segmentInfo attributes so that we can read it when loading the
        // codec from SPI.
        // This is similar to whats done in
        // https://github.com/opensearch-project/custom-codecs/blob/2.19.0.0/src/main/java/org/opensearch/index/codec/customcodecs/Lucene912CustomStoredFieldsFormat.java#L95-L100
        String previous = segmentInfo.putAttribute(DELEGATE_CODEC_KEY, delegateCodecName);
        if (previous != null && previous.equals(delegateCodecName) == false) {
            throw new IllegalStateException(
                "found existing value for "
                    + DELEGATE_CODEC_KEY
                    + " for segment: "
                    + segmentInfo.name
                    + " old = "
                    + previous
                    + ", new = "
                    + delegateCodecName
            );
        }

        StoredFieldsWriter delegateWriter = delegate.fieldsWriter(directory, segmentInfo, ioContext);
        if (mapperService != null && KNNSettings.isKNNDerivedSourceEnabled(mapperService.getIndexSettings().getSettings())) {
            List<String> vectorFieldTypes = new ArrayList<>();
            for (MappedFieldType fieldType : mapperService.fieldTypes()) {
                if (fieldType instanceof KNNVectorFieldType) {
                    vectorFieldTypes.add(fieldType.name());
                }
            }
            if (vectorFieldTypes.isEmpty() == false) {
                return new DerivedSourceStoredFieldsWriter(delegateWriter, vectorFieldTypes);
            }
        }
        return delegateWriter;
    }
}

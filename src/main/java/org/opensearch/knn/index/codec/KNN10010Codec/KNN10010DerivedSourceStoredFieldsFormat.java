/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN10010Codec;

import com.google.common.annotations.VisibleForTesting;
import lombok.AllArgsConstructor;
import lombok.extern.log4j.Log4j2;

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
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.derivedsource.DerivedFieldInfo;
import org.opensearch.knn.index.codec.derivedsource.DerivedSourceReaders;
import org.opensearch.knn.index.codec.derivedsource.DerivedSourceReadersSupplier;
import org.opensearch.knn.index.codec.derivedsource.DerivedSourceSegmentAttributeParser;
import org.opensearch.knn.index.util.IndexUtil;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@AllArgsConstructor
@Log4j2
public class KNN10010DerivedSourceStoredFieldsFormat extends StoredFieldsFormat {
    // Stores the delegate codec name (in case it is different from the default one)
    static final String KNN_DELEGATE_CODEC_NAME = "knn_delegate_stored_fields_codec_key";

    private final String name;
    private final StoredFieldsFormat delegate;
    private final DerivedSourceReadersSupplier derivedSourceReadersSupplier;
    // IMPORTANT Do not rely on this for the reader, it will be null if SPI is used
    @Nullable
    private final MapperService mapperService;

    @Override
    public StoredFieldsReader fieldsReader(Directory directory, SegmentInfo segmentInfo, FieldInfos fieldInfos, IOContext ioContext)
        throws IOException {

        final StoredFieldsFormat delegatingFormat = getStoredFieldsFormat(segmentInfo);
        List<DerivedFieldInfo> derivedVectorFields = new ArrayList<>();
        for (String field : DerivedSourceSegmentAttributeParser.parseDerivedVectorFields(segmentInfo, false)) {
            addDerivedFieldInfo(derivedVectorFields, fieldInfos, field, false);
        }
        for (String field : DerivedSourceSegmentAttributeParser.parseDerivedVectorFields(segmentInfo, true)) {
            addDerivedFieldInfo(derivedVectorFields, fieldInfos, field, true);
        }

        // If no fields have it enabled, we can just short-circuit and return the delegate's fieldReader
        if (derivedVectorFields.isEmpty()) {
            return delegatingFormat.fieldsReader(directory, segmentInfo, fieldInfos, ioContext);
        }
        SegmentReadState segmentReadState = new SegmentReadState(directory, segmentInfo, fieldInfos, ioContext);
        DerivedSourceReaders derivedSourceReaders = derivedSourceReadersSupplier.getReaders(segmentReadState);
        return new KNN10010DerivedSourceStoredFieldsReader(
            delegatingFormat.fieldsReader(directory, segmentInfo, fieldInfos, ioContext),
            derivedVectorFields,
            derivedSourceReaders,
            segmentReadState
        );
    }

    private StoredFieldsFormat getStoredFieldsFormat(final SegmentInfo segmentInfo) throws IOException {
        // Apache Lucene does not have an SPI for StoredFieldsFormat, so we are doing Codec lookups
        final String name = segmentInfo.getAttribute(KNN_DELEGATE_CODEC_NAME);
        if (name != null && !this.name.equalsIgnoreCase(name)) {
            // Only when name is different from the default one (delegate)
            return Codec.forName(name).storedFieldsFormat();
        } else {
            return delegate;
        }
    }

    private void addDerivedFieldInfo(List<DerivedFieldInfo> derivedFieldInfos, FieldInfos fieldInfos, String field, boolean isNested) {
        FieldInfo fieldInfo = getFieldInfo(fieldInfos, field);
        if (fieldInfo == null) {
            return;
        }
        derivedFieldInfos.add(new DerivedFieldInfo(fieldInfo, isNested));
    }

    @VisibleForTesting
    static FieldInfo getFieldInfo(FieldInfos fieldInfos, String field) {
        FieldInfo fieldInfo = fieldInfos.fieldInfo(field);
        if (fieldInfo != null) {
            return fieldInfo;
        }

        FieldInfo matchedFieldInfo = null;
        for (FieldInfo candidate : fieldInfos) {
            if (candidate.name.equalsIgnoreCase(field) == false) {
                continue;
            }
            if (matchedFieldInfo == null) {
                matchedFieldInfo = candidate;
                continue;
            }

            boolean matchedHasVectorValues = matchedFieldInfo.hasVectorValues();
            boolean candidateHasVectorValues = candidate.hasVectorValues();
            if (matchedHasVectorValues == candidateHasVectorValues) {
                log.warn(
                    "Skipping derived vector field [{}] because field infos [{}] and [{}] both match case-insensitively",
                    field,
                    matchedFieldInfo.name,
                    candidate.name
                );
                return null;
            }
            if (candidateHasVectorValues) {
                matchedFieldInfo = candidate;
            }
        }
        return matchedFieldInfo;
    }

    @Override
    public StoredFieldsWriter fieldsWriter(Directory directory, SegmentInfo segmentInfo, IOContext ioContext) throws IOException {
        // Store delegate codec name to be used by reader side
        String previous = segmentInfo.putAttribute(KNN_DELEGATE_CODEC_NAME, name);
        if (previous != null && previous.equals(name) == false) {
            throw new IllegalStateException(
                "Found existing value for "
                    + KNN_DELEGATE_CODEC_NAME
                    + " for segment: "
                    + segmentInfo.name
                    + " old = "
                    + previous
                    + ", new = "
                    + name
            );
        }

        StoredFieldsWriter delegateWriter = delegate.fieldsWriter(directory, segmentInfo, ioContext);
        if (IndexUtil.isDerivedEnabledForIndex(mapperService) == false) {
            return delegateWriter;
        }

        // Just pass mapperService - we'll query for fields in finish() when all mappings exist
        return new KNN10010DerivedSourceStoredFieldsWriter(name, delegateWriter, segmentInfo, mapperService);
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN10010Codec;

import lombok.AllArgsConstructor;
import org.apache.lucene.codecs.StoredFieldsFormat;
import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.codecs.StoredFieldsWriter;
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
import java.util.List;
import java.util.stream.Stream;

@AllArgsConstructor
public class KNN10010DerivedSourceStoredFieldsFormat extends StoredFieldsFormat {

    private final StoredFieldsFormat delegate;
    private final DerivedSourceReadersSupplier derivedSourceReadersSupplier;
    // IMPORTANT Do not rely on this for the reader, it will be null if SPI is used
    @Nullable
    private final MapperService mapperService;

    @Override
    public StoredFieldsReader fieldsReader(Directory directory, SegmentInfo segmentInfo, FieldInfos fieldInfos, IOContext ioContext)
        throws IOException {
        List<DerivedFieldInfo> derivedVectorFields = Stream.concat(
            DerivedSourceSegmentAttributeParser.parseDerivedVectorFields(segmentInfo, false)
                .stream()
                .filter(field -> fieldInfos.fieldInfo(field) != null)
                .map(field -> new DerivedFieldInfo(fieldInfos.fieldInfo(field), false)),
            DerivedSourceSegmentAttributeParser.parseDerivedVectorFields(segmentInfo, true)
                .stream()
                .filter(field -> fieldInfos.fieldInfo(field) != null)
                .map(field -> new DerivedFieldInfo(fieldInfos.fieldInfo(field), true))
        ).toList();

        // If no fields have it enabled, we can just short-circuit and return the delegate's fieldReader
        if (derivedVectorFields.isEmpty()) {
            return delegate.fieldsReader(directory, segmentInfo, fieldInfos, ioContext);
        }
        SegmentReadState segmentReadState = new SegmentReadState(directory, segmentInfo, fieldInfos, ioContext);
        DerivedSourceReaders derivedSourceReaders = derivedSourceReadersSupplier.getReaders(segmentReadState);
        return new KNN10010DerivedSourceStoredFieldsReader(
            delegate.fieldsReader(directory, segmentInfo, fieldInfos, ioContext),
            derivedVectorFields,
            derivedSourceReaders,
            segmentReadState
        );
    }

    @Override
    public StoredFieldsWriter fieldsWriter(Directory directory, SegmentInfo segmentInfo, IOContext ioContext) throws IOException {
        StoredFieldsWriter delegateWriter = delegate.fieldsWriter(directory, segmentInfo, ioContext);
        if (IndexUtil.isDerivedEnabledForIndex(mapperService) == false) {
            return delegateWriter;
        }

        // Just pass mapperService - we'll query for fields in finish() when all mappings exist
        return new KNN10010DerivedSourceStoredFieldsWriter(delegateWriter, segmentInfo, mapperService);
    }
}

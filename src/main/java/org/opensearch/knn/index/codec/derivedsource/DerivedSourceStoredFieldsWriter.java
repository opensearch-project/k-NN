/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import lombok.RequiredArgsConstructor;
import org.apache.lucene.codecs.StoredFieldsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.store.DataInput;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.collect.Tuple;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.common.xcontent.support.XContentMapValues;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.MediaType;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.mapper.SourceFieldMapper;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Map;
import java.util.Objects;

@RequiredArgsConstructor
public class DerivedSourceStoredFieldsWriter extends StoredFieldsWriter {

    private final StoredFieldsWriter delegate;
    private final List<String> vectorFieldTypes;

    @Override
    public void startDocument() throws IOException {
        delegate.startDocument();
    }

    @Override
    public void writeField(FieldInfo fieldInfo, int i) throws IOException {
        delegate.writeField(fieldInfo, i);
    }

    @Override
    public void writeField(FieldInfo fieldInfo, long l) throws IOException {
        delegate.writeField(fieldInfo, l);
    }

    @Override
    public void writeField(FieldInfo fieldInfo, float v) throws IOException {
        delegate.writeField(fieldInfo, v);
    }

    @Override
    public void writeField(FieldInfo fieldInfo, double v) throws IOException {
        delegate.writeField(fieldInfo, v);
    }

    @Override
    public void writeField(FieldInfo info, DataInput value, int length) throws IOException {
        delegate.writeField(info, value, length);
    }

    @Override
    public int merge(MergeState mergeState) throws IOException {
        // We have to wrap these here to avoid storing the vectors during merge
        for (int i = 0; i < mergeState.storedFieldsReaders.length; i++) {
            mergeState.storedFieldsReaders[i] = DerivedSourceStoredFieldsReader.wrapForMerge(mergeState.storedFieldsReaders[i]);
        }
        return delegate.merge(mergeState);
    }

    @Override
    public void writeField(FieldInfo fieldInfo, BytesRef bytesRef) throws IOException {
        // Parse out the vectors from the source
        if (Objects.equals(fieldInfo.name, SourceFieldMapper.NAME) && !vectorFieldTypes.isEmpty()) {
            // Reference:
            // https://github.com/opensearch-project/OpenSearch/blob/2.18.0/server/src/main/java/org/opensearch/index/mapper/SourceFieldMapper.java#L322
            Tuple<? extends MediaType, Map<String, Object>> mapTuple = XContentHelper.convertToMap(
                BytesReference.fromByteBuffer(ByteBuffer.wrap(bytesRef.bytes, bytesRef.offset, bytesRef.length)),
                true,
                MediaTypeRegistry.JSON
            );
            Map<String, Object> filteredSource = XContentMapValues.filter(null, vectorFieldTypes.toArray(new String[0]))
                .apply(mapTuple.v2());
            BytesStreamOutput bStream = new BytesStreamOutput();
            MediaType actualContentType = mapTuple.v1();
            XContentBuilder builder = MediaTypeRegistry.contentBuilder(actualContentType, bStream).map(filteredSource);
            builder.close();
            BytesReference bytesReference = bStream.bytes();
            delegate.writeField(fieldInfo, bytesReference.toBytesRef());
            return;
        }
        delegate.writeField(fieldInfo, bytesRef);
    }

    @Override
    public void writeField(FieldInfo fieldInfo, String s) throws IOException {
        delegate.writeField(fieldInfo, s);
    }

    @Override
    public void finishDocument() throws IOException {
        delegate.finishDocument();
    }

    @Override
    public void finish(int i) throws IOException {
        delegate.finish(i);
    }

    @Override
    public void close() throws IOException {
        delegate.close();
    }

    @Override
    public long ramBytesUsed() {
        return delegate.ramBytesUsed();
    }
}

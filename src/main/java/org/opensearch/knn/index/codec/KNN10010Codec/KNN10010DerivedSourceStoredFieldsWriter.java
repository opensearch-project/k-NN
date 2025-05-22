/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN10010Codec;

import org.apache.lucene.codecs.StoredFieldsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.StoredFieldDataInput;
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
import org.opensearch.knn.index.codec.backward_codecs.KNN9120Codec.KNN9120DerivedSourceStoredFieldsReader;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;

public class KNN10010DerivedSourceStoredFieldsWriter extends StoredFieldsWriter {

    private final StoredFieldsWriter delegate;
    private final Function<Map<String, Object>, Map<String, Object>> vectorMask;

    // Keeping the mask as small as possible.
    public final static Byte MASK = 0x1;

    /**
     *
     * @param delegate StoredFieldsWriter to wrap
     * @param vectorFieldTypesArg List of vector field types to mask. If empty, no masking will be done
     */
    public KNN10010DerivedSourceStoredFieldsWriter(StoredFieldsWriter delegate, List<String> vectorFieldTypesArg) {
        this.delegate = delegate;
        List<String> vectorFieldTypes = vectorFieldTypesArg.stream().map(String::toLowerCase).toList();
        if (vectorFieldTypes.isEmpty() == false) {
            this.vectorMask = XContentMapValues.transform(
                vectorFieldTypes.stream().collect(Collectors.toMap(k -> k, k -> (Object o) -> o == null ? o : MASK)),
                true
            );
        } else {
            this.vectorMask = null;
        }
    }

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
    public void writeField(FieldInfo info, StoredFieldDataInput value) throws IOException {
        delegate.writeField(info, value);
    }

    @Override
    public int merge(MergeState mergeState) throws IOException {
        // In case of backwards compatibility, with old segments, we need to perform a non-optimal merge. Basically, it
        // will repopulate each source and then inject the vector and then remove it. This allows us to migrate
        // segments from filter approach to mask approach
        if (KNN9120DerivedSourceStoredFieldsReader.doesMergeContainLegacySegments(mergeState)) {
            return super.merge(mergeState);
        }

        // We wrap the segments to avoid injecting back vectors and then removing. If this is not done, then we will
        // inject and then just write to disk potentially.
        for (int i = 0; i < mergeState.storedFieldsReaders.length; i++) {
            mergeState.storedFieldsReaders[i] = KNN10010DerivedSourceStoredFieldsReader.wrapForMerge(mergeState.storedFieldsReaders[i]);
        }
        return delegate.merge(mergeState);
    }

    @Override
    public void writeField(FieldInfo fieldInfo, BytesRef bytesRef) throws IOException {
        // Parse out the vectors from the source
        if (vectorMask != null && Objects.equals(fieldInfo.name, SourceFieldMapper.NAME)) {
            // Reference:
            // https://github.com/opensearch-project/OpenSearch/blob/2.18.0/server/src/main/java/org/opensearch/index/mapper/SourceFieldMapper.java#L322

            // In the Index operation listener, preindex, we apply the mask operation. We apply it again for safety
            // here because writeField may not always follow the preindex step, so it might have the vector (think
            // merge). This may be overly cautious and we can remove/optimize it in the future. For now, its a safety
            // net.
            Tuple<? extends MediaType, Map<String, Object>> mapTuple = XContentHelper.convertToMap(
                BytesReference.fromByteBuffer(ByteBuffer.wrap(bytesRef.bytes, bytesRef.offset, bytesRef.length)),
                true,
                MediaTypeRegistry.JSON
            );
            Map<String, Object> filteredSource = vectorMask.apply(mapTuple.v2());
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

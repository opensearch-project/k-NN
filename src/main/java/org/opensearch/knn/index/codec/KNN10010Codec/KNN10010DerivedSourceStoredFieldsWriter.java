/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN10010Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.codecs.StoredFieldsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.StoredFieldDataInput;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.collect.Tuple;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.common.xcontent.support.XContentMapValues;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.compress.NotXContentException;
import org.opensearch.core.xcontent.MediaType;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.index.mapper.SourceFieldMapper;
import org.opensearch.knn.index.codec.backward_codecs.KNN9120Codec.KNN9120DerivedSourceStoredFieldsReader;
import org.opensearch.knn.index.codec.derivedsource.DerivedFieldInfo;
import org.opensearch.knn.index.codec.derivedsource.DerivedSourceSegmentAttributeParser;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;
import org.opensearch.knn.index.util.IndexUtil;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

@Log4j2
public class KNN10010DerivedSourceStoredFieldsWriter extends StoredFieldsWriter {

    private final StoredFieldsWriter delegate;
    private final SegmentInfo segmentInfo;
    private final MapperService mapperService;

    private final Set<String> knownVectorFields = new HashSet<>();

    private Function<Map<String, Object>, Map<String, Object>> vectorMask;

    // Keeping the mask as small as possible.
    public final static Byte MASK = 0x1;

    /**
     *
     * @param delegate StoredFieldsWriter to wrap
     */
    public KNN10010DerivedSourceStoredFieldsWriter(StoredFieldsWriter delegate, SegmentInfo segmentInfo, MapperService mapperService) {
        this.delegate = delegate;
        this.segmentInfo = segmentInfo;
        this.mapperService = mapperService;

        // Initialize with currently known fields
        // The vector mask will be initialized here upon the first document in the segment
        // Will work like an incremental cache, updating the vector mask as new fields are discovered
        for (MappedFieldType fieldType : mapperService.fieldTypes()) {
            if (fieldType instanceof KNNVectorFieldType knnVectorFieldType) {
                if (IndexUtil.isDerivedEnabledForField(knnVectorFieldType, mapperService)) {
                    knownVectorFields.add(fieldType.name().toLowerCase());
                }
            }
        }

        if (!knownVectorFields.isEmpty()) {
            vectorMask = buildMask(knownVectorFields);
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

    /**
     * Builds a mask transform function for the given fields.
     */
    private Function<Map<String, Object>, Map<String, Object>> buildMask(Set<String> fields) {
        return XContentMapValues.transform(
            fields.stream().collect(Collectors.toMap(k -> k, k -> (Object o) -> o == null ? o : MASK)),
            true
        );
    }

    /**
     * Quick check if source contains any unmasked vectors.
     * Returns false if all vectors are already masked (value = 1) or no vectors present.
     */
    private boolean containsUnmaskedVectors(Map<String, Object> source) {
        for (Map.Entry<String, Object> entry : source.entrySet()) {
            Object value = entry.getValue();

            // Check if this is an unmasked vector (List of Numbers with more than 1 element)
            if (value instanceof List<?> list && !list.isEmpty()) {
                Object first = list.get(0);
                if (first instanceof Number && list.size() > 1) {
                    return true;
                }
            }

            // Recurse into nested objects
            if (value instanceof Map) {
                @SuppressWarnings("unchecked")
                Map<String, Object> nested = (Map<String, Object>) value;
                if (containsUnmaskedVectors(nested)) {
                    return true;
                }
            }

            // Recurse into arrays of objects
            if (value instanceof List<?> list) {
                for (Object item : list) {
                    if (item instanceof Map) {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> nestedItem = (Map<String, Object>) item;
                        if (containsUnmaskedVectors(nestedItem)) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }

    /**
     * Recursively find KNN vector fields in document source.
     * Only called in slow path when unmasked vectors are detected.
     */
    private Set<String> findKnnVectorFields(Map<String, Object> source, String prefix) {
        Set<String> fields = new HashSet<>();
        for (Map.Entry<String, Object> entry : source.entrySet()) {
            String fullPath = prefix.isEmpty() ? entry.getKey() : prefix + "." + entry.getKey();

            MappedFieldType fieldType = mapperService.fieldType(fullPath);
            if (fieldType instanceof KNNVectorFieldType knnVectorFieldType) {
                if (IndexUtil.isDerivedEnabledForField(knnVectorFieldType, mapperService)) {
                    fields.add(fullPath.toLowerCase());
                }
            }

            Object value = entry.getValue();
            if (value instanceof Map) {
                @SuppressWarnings("unchecked")
                Map<String, Object> nested = (Map<String, Object>) value;
                fields.addAll(findKnnVectorFields(nested, fullPath));
            } else if (value instanceof List<?> list) {
                for (Object item : list) {
                    if (item instanceof Map) {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> nestedItem = (Map<String, Object>) item;
                        fields.addAll(findKnnVectorFields(nestedItem, fullPath));
                    }
                }
            }
        }
        return fields;
    }

    /**
     * During merge, multiple source segments are combined into a single new segment.
     * The merged segment must know about ALL derived vector fields from ALL source segments.
     *
     * We have to write segment attributes here as well as in finish because during
     * merge operations, {@code delegate.merge()} internally calls the delegate's finish(),
     * not ours. Therefore, we must write attributes directly after the merge completes.
     *
     * @param mergeState contains readers for source segments and info for the target segment
     * @return the number of documents merged
     * @throws IOException if an I/O error occurs during merge
     */
    @Override
    public int merge(MergeState mergeState) throws IOException {
        if (KNN9120DerivedSourceStoredFieldsReader.doesMergeContainLegacySegments(mergeState)) {
            return super.merge(mergeState);
        }

        Set<String> vectorFields = new HashSet<>();
        Set<String> nestedVectorFields = new HashSet<>();

        for (int i = 0; i < mergeState.storedFieldsReaders.length; i++) {
            StoredFieldsReader reader = mergeState.storedFieldsReaders[i];

            if (reader instanceof KNN10010DerivedSourceStoredFieldsReader knnReader) {
                for (DerivedFieldInfo fieldInfo : knnReader.getDerivedVectorFields()) {
                    if (fieldInfo.isNested()) {
                        nestedVectorFields.add(fieldInfo.name());
                    } else {
                        vectorFields.add(fieldInfo.name());
                    }
                }
            }
        }

        for (int i = 0; i < mergeState.storedFieldsReaders.length; i++) {
            mergeState.storedFieldsReaders[i] = KNN10010DerivedSourceStoredFieldsReader.wrapForMerge(mergeState.storedFieldsReaders[i]);
        }

        int result = delegate.merge(mergeState);

        if (!vectorFields.isEmpty()) {
            List<String> sortedVectorFields = new ArrayList<>(vectorFields);
            Collections.sort(sortedVectorFields);
            DerivedSourceSegmentAttributeParser.addDerivedVectorFieldsSegmentInfoAttribute(
                segmentInfo,
                new ArrayList<>(vectorFields),
                false
            );
        }
        if (!nestedVectorFields.isEmpty()) {
            List<String> sortedNestedVectorFields = new ArrayList<>(nestedVectorFields);
            Collections.sort(sortedNestedVectorFields);
            DerivedSourceSegmentAttributeParser.addDerivedVectorFieldsSegmentInfoAttribute(
                segmentInfo,
                new ArrayList<>(nestedVectorFields),
                true
            );
        }

        return result;
    }

    @Override
    public void writeField(FieldInfo fieldInfo, BytesRef bytesRef) throws IOException {
        if (!Objects.equals(fieldInfo.name, SourceFieldMapper.NAME)) {
            delegate.writeField(fieldInfo, bytesRef);
            return;
        }

        // Parse _source
        Tuple<? extends MediaType, Map<String, Object>> mapTuple;
        try {
            mapTuple = XContentHelper.convertToMap(
                BytesReference.fromByteBuffer(ByteBuffer.wrap(bytesRef.bytes, bytesRef.offset, bytesRef.length)),
                true,
                MediaTypeRegistry.JSON
            );
        } catch (NotXContentException e) {
            log.warn(
                "Encountered NotXContent while deserializing _source field. Instead found String: [{}]",
                new String(bytesRef.bytes, 0, Math.min(bytesRef.bytes.length, 512)),
                e
            );
            return;
        }

        // Check if source is already masked from preindex masking path.
        // If it is, we do not need to remask the source here
        if (!containsUnmaskedVectors(mapTuple.v2())) {
            delegate.writeField(fieldInfo, bytesRef);
            return;
        }

        // This path is only if source is not masked from preindex path
        // Eg: preindex failed, merging legacy segment with modern segment - need to mask
        // Find KNN fields in this document
        if (vectorMask != null) {
            Map<String, Object> filteredSource = vectorMask.apply(mapTuple.v2());

            // Check if existing masking worked (no unmasked vectors remain)
            if (!containsUnmaskedVectors(filteredSource)) {
                // Cached mask was sufficient!
                BytesStreamOutput bStream = new BytesStreamOutput();
                MediaType actualContentType = mapTuple.v1();
                XContentBuilder builder = MediaTypeRegistry.contentBuilder(actualContentType, bStream).map(filteredSource);
                builder.close();
                delegate.writeField(fieldInfo, bStream.bytes().toBytesRef());
                return;
            }
        }

        // Cached mask didn't cover all fields - need to discover new ones and build new mask
        Set<String> fieldsInDoc = findKnnVectorFields(mapTuple.v2(), "");
        knownVectorFields.addAll(fieldsInDoc);
        vectorMask = buildMask(knownVectorFields);

        // Apply mask if we have one
        if (vectorMask != null) {
            Map<String, Object> filteredSource = vectorMask.apply(mapTuple.v2());
            BytesStreamOutput bStream = new BytesStreamOutput();
            MediaType actualContentType = mapTuple.v1();
            XContentBuilder builder = MediaTypeRegistry.contentBuilder(actualContentType, bStream).map(filteredSource);
            builder.close();
            delegate.writeField(fieldInfo, bStream.bytes().toBytesRef());
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

    /**
     * Writes derived vector field attributes after the segment writing is finished
     * Previously, segment attributes were written in {@code fieldsWriter()}
     * at segment creation time. When bulk indexing documents with dynamic templates that create
     * new field mappings, only the
     * first document's mapping existed at that point. Subsequent documents' dynamic mappings
     * were created as they were parsed, AFTER the segment attributes were already written.
     *
     * This caused the segment attribute to be incomplete:
     * By moving segment attribute writing to {@code finish()}, we ensure all
     * documents have been parsed and all dynamic mappings exist in {@code mapperService}
     * before we query for the complete list of derived vector fields.
     *
     * @param i the number of documents in the segment
     * @throws IOException if an I/O error occurs
     */
    @Override
    public void finish(int i) throws IOException {
        List<String> vectorFieldTypes;
        List<String> nestedVectorFieldTypes;

        vectorFieldTypes = new ArrayList<>();
        nestedVectorFieldTypes = new ArrayList<>();

        for (MappedFieldType fieldType : mapperService.fieldTypes()) {
            if (fieldType instanceof KNNVectorFieldType knnVectorFieldType) {
                if (IndexUtil.isDerivedEnabledForField(knnVectorFieldType, mapperService) == false) {
                    continue;
                }

                boolean isNested = mapperService.documentMapper().mappers().getNestedScope(fieldType.name()) != null;
                if (isNested) {
                    nestedVectorFieldTypes.add(fieldType.name());
                } else {
                    vectorFieldTypes.add(fieldType.name());
                }
            }
        }

        // Write segment attributes
        if (!vectorFieldTypes.isEmpty()) {
            Collections.sort(vectorFieldTypes);
            DerivedSourceSegmentAttributeParser.addDerivedVectorFieldsSegmentInfoAttribute(segmentInfo, vectorFieldTypes, false);
        }
        if (!nestedVectorFieldTypes.isEmpty()) {
            Collections.sort(nestedVectorFieldTypes);
            DerivedSourceSegmentAttributeParser.addDerivedVectorFieldsSegmentInfoAttribute(segmentInfo, nestedVectorFieldTypes, true);
        }

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

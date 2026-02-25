/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.SegmentReadState;
import org.opensearch.common.collect.Tuple;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.common.xcontent.support.XContentMapValues;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.MediaType;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.regex.Regex;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.HashSet;
import java.util.Set;

@Log4j2
public class DerivedSourceVectorTransformer {

    private final DerivedSourceReaders derivedSourceReaders;
    Function<Map<String, Object>, Map<String, Object>> derivedSourceVectorTransformer;
    Map<String, PerFieldDerivedVectorTransformer> perFieldDerivedVectorTransformers;
    private boolean isNested;
    private final DerivedSourceLuceneHelper derivedSourceLuceneHelper;

    /**
     *
     * @param derivedSourceReaders derived source readers.
     * @param segmentReadState Segment read state
     * @param fieldsToInjectVector List of fields to inject vectors into
     */
    public DerivedSourceVectorTransformer(
        DerivedSourceReaders derivedSourceReaders,
        SegmentReadState segmentReadState,
        List<DerivedFieldInfo> fieldsToInjectVector
    ) {
        this.derivedSourceReaders = derivedSourceReaders;
        perFieldDerivedVectorTransformers = new HashMap<>();
        for (DerivedFieldInfo derivedFieldInfo : fieldsToInjectVector) {
            isNested = derivedFieldInfo.isNested() || isNested;
            PerFieldDerivedVectorTransformer perFieldDerivedVectorTransformer = PerFieldDerivedVectorTransformerFactory.create(
                derivedFieldInfo.fieldInfo(),
                derivedFieldInfo.isNested(),
                derivedSourceReaders
            );
            perFieldDerivedVectorTransformers.put(derivedFieldInfo.name(), perFieldDerivedVectorTransformer);
        }
        derivedSourceLuceneHelper = new DerivedSourceLuceneHelper(derivedSourceReaders, segmentReadState);
    }

    /**
     * Initialize the transformer with the fields that should be injected based on includes/excludes.
     * This filters perFieldDerivedVectorTransformers and builds the derivedSourceVectorTransformer.
     * Should be called once before the first call to injectVectors().
     *
     * @param excludes List of field patterns that should not be injected
     */
    public void initialize(String[] includes, String[] excludes) {
        // Filter perFieldDerivedVectorTransformers based on includes/excludes
        Set<String> fieldsToRemove = getFieldsToExclude(includes, excludes);
        for (String fieldName : fieldsToRemove) {
            perFieldDerivedVectorTransformers.remove(fieldName);
        }

        Map<String, Function<Object, Object>> transformerFunctions = new HashMap<>();
        transformerFunctions.putAll(perFieldDerivedVectorTransformers);
        derivedSourceVectorTransformer = XContentMapValues.transform(transformerFunctions, true);

    }

    private Set<String> getFieldsToExclude(String[] includes, String[] excludes) {
        Set<String> fieldsToRemove = new HashSet<>();
        Set<String> allFields = perFieldDerivedVectorTransformers.keySet();

        // If includes are specified, start by excluding everything that doesn't match
        if (includes != null && includes.length > 0) {
            for (String fieldName : allFields) {
                boolean matched = false;
                for (String includePattern : includes) {
                    if (Regex.simpleMatch(includePattern, fieldName)) {
                        matched = true;
                        break;
                    }
                }
                if (!matched) {
                    fieldsToRemove.add(fieldName);
                }
            }
        }

        // Then also exclude anything matching exclude patterns
        if (excludes != null && excludes.length > 0) {
            for (String fieldName : allFields) {
                if (fieldsToRemove.contains(fieldName)) {
                    continue;
                }
                for (String excludePattern : excludes) {
                    if (Regex.simpleMatch(excludePattern, fieldName)) {
                        fieldsToRemove.add(fieldName);
                        break;
                    }
                }
            }
        }

        return fieldsToRemove;
    }

    /**
     * Check if there are any fields to inject after initialization.
     * Must be called after initialize().
     *
     * @return true if there are fields to inject, false otherwise
     */
    public boolean hasFieldsToInject() {
        return !perFieldDerivedVectorTransformers.isEmpty();
    }

    /**
     * Given a docId and the source of that doc as bytes, add all the necessary vector fields into the source.
     *
     * @param docId doc id of the document
     * @param sourceAsBytes source of document as bytes
     * @return byte array of the source with the vector fields added
     * @throws IOException if there is an issue reading from the formats
     */
    public byte[] injectVectors(int docId, byte[] sourceAsBytes) throws IOException {
        // Reference:
        // https://github.com/opensearch-project/OpenSearch/blob/2.18.0/server/src/main/java/org/opensearch/index/mapper/SourceFieldMapper.java#L322
        // Deserialize the source into a modifiable map
        Tuple<? extends MediaType, Map<String, Object>> mapTuple = XContentHelper.convertToMap(
            BytesReference.fromByteBuffer(ByteBuffer.wrap(sourceAsBytes)),
            true,
            MediaTypeRegistry.getDefaultMediaType()
        );
        // Have to create a copy of the map here to ensure that is mutable
        Map<String, Object> sourceAsMap = mapTuple.v2();

        // We only need the offset for the nested fields. If there arent any, we can skip
        int offset = 0;
        if (isNested) {
            offset = derivedSourceLuceneHelper.getFirstChild(docId);
        }

        // For each vector field, add in the source. The per field injectors are responsible for skipping if
        // the field is not present.
        for (PerFieldDerivedVectorTransformer vectorTransformer : perFieldDerivedVectorTransformers.values()) {
            vectorTransformer.setCurrentDoc(offset, docId);
        }

        Map<String, Object> copy = derivedSourceVectorTransformer.apply(sourceAsMap);

        // At this point, we can serialize the modified source map
        // Setting to 1024 based on
        // https://github.com/opensearch-project/OpenSearch/blob/2.18.0/server/src/main/java/org/opensearch/search/fetch/subphase/FetchSourcePhase.java#L106
        BytesStreamOutput bStream = new BytesStreamOutput(1024);
        MediaType actualContentType = mapTuple.v1();
        XContentBuilder builder = MediaTypeRegistry.contentBuilder(actualContentType, bStream).map(copy);
        builder.close();
        return BytesReference.toBytes(BytesReference.bytes(builder));
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.backward_codecs.KNN9120Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentReadState;
import org.opensearch.common.collect.Tuple;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.core.common.Strings;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.MediaType;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * This class is responsible for injecting vectors into the source of a document. From a high level, it uses alternative
 *  format readers and information about the fields to inject vectors into the source.
 */
@Log4j2
public class DerivedSourceVectorInjector {

    private final KNN9120DerivedSourceReaders derivedSourceReaders;
    private final List<PerFieldDerivedVectorInjector> perFieldDerivedVectorInjectors;
    private final Set<String> fieldNames;

    /**
     * Constructor for DerivedSourceVectorInjector.
     *
     * @param derivedSourceReaders Derived source readers.
     * @param segmentReadState Segment read state
     * @param fieldsToInjectVector List of fields to inject vectors into
     */
    public DerivedSourceVectorInjector(
        KNN9120DerivedSourceReaders derivedSourceReaders,
        SegmentReadState segmentReadState,
        List<FieldInfo> fieldsToInjectVector
    ) {
        this.derivedSourceReaders = derivedSourceReaders;
        this.perFieldDerivedVectorInjectors = new ArrayList<>();
        this.fieldNames = new HashSet<>();
        for (FieldInfo fieldInfo : fieldsToInjectVector) {
            this.perFieldDerivedVectorInjectors.add(
                PerFieldDerivedVectorInjectorFactory.create(fieldInfo, derivedSourceReaders, segmentReadState)
            );
            this.fieldNames.add(fieldInfo.name);
        }
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
        Map<String, Object> sourceAsMap = new HashMap<>(mapTuple.v2());

        // For each vector field, add in the source. The per field injectors are responsible for skipping if
        // the field is not present.
        for (PerFieldDerivedVectorInjector vectorInjector : perFieldDerivedVectorInjectors) {
            vectorInjector.inject(docId, sourceAsMap);
        }

        // At this point, we can serialize the modified source map
        // Setting to 1024 based on
        // https://github.com/opensearch-project/OpenSearch/blob/2.18.0/server/src/main/java/org/opensearch/search/fetch/subphase/FetchSourcePhase.java#L106
        BytesStreamOutput bStream = new BytesStreamOutput(1024);
        MediaType actualContentType = mapTuple.v1();
        XContentBuilder builder = MediaTypeRegistry.contentBuilder(actualContentType, bStream).map(sourceAsMap);
        builder.close();
        return BytesReference.toBytes(BytesReference.bytes(builder));
    }

    /**
     * Whether or not to inject vectors based on what fields are explicitly required
     *
     * @param includes List of fields that are required to be injected
     * @param excludes List of fields that are not required to be injected
     * @return true if vectors should be injected, false otherwise
     */
    public boolean shouldInject(String[] includes, String[] excludes) {
        // If any of the vector fields are explicitly required we should inject
        if (includes != null && includes != Strings.EMPTY_ARRAY) {
            for (String includedField : includes) {
                if (fieldNames.contains(includedField)) {
                    return true;
                }
            }
        }

        // If all of the vector fields are explicitly excluded we should not inject
        if (excludes != null && excludes != Strings.EMPTY_ARRAY) {
            int excludedVectorFieldCount = 0;
            for (String excludedField : excludes) {
                if (fieldNames.contains(excludedField)) {
                    excludedVectorFieldCount++;
                }
            }
            // Inject if we havent excluded all of the fields
            return excludedVectorFieldCount < fieldNames.size();
        }
        return true;
    }
}

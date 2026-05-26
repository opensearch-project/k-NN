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

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

@Log4j2
public class DerivedSourceVectorTransformer {

    private final DerivedSourceReaders derivedSourceReaders;
    private final Function<Map<String, Object>, Map<String, Object>> derivedSourceVectorTransformer;
    private final Map<String, Function<Object, Object>> perFieldDerivedVectorTransformers;
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
        derivedSourceVectorTransformer = XContentMapValues.transform(perFieldDerivedVectorTransformers, true);
        derivedSourceLuceneHelper = new DerivedSourceLuceneHelper(derivedSourceReaders, segmentReadState);
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
        for (final Function<Object, Object> vectorTransformer : perFieldDerivedVectorTransformers.values()) {
            ((PerFieldDerivedVectorTransformer) vectorTransformer).setCurrentDoc(offset, docId);
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

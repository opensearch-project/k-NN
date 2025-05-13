/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.collect.Tuple;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.common.xcontent.support.XContentMapValues;
import org.opensearch.core.index.shard.ShardId;
import org.opensearch.core.xcontent.MediaType;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.index.mapper.SourceFieldMapper;
import org.opensearch.index.shard.IndexingOperationListener;
import org.opensearch.knn.index.DerivedKnnByteVectorField;
import org.opensearch.knn.index.DerivedKnnFloatVectorField;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNN10010Codec.KNN10010DerivedSourceStoredFieldsWriter;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Function;

import static org.opensearch.index.mapper.SourceFieldMapper.RECOVERY_SOURCE_NAME;

/**
 * Before applying the indexing operation, we need to ensure that the source that gets added to the translog matches
 * exactly what we will reconstruct. To do this, we reconstruct the source from the binary source and then apply the
 * transformation on top of it and then set the source back.
 */
@Log4j2
public class DerivedSourceIndexOperationListener implements IndexingOperationListener {

    @Override
    public Engine.Index preIndex(ShardId shardId, Engine.Index operation) {
        // If recovery source is enabled, we do not need to modify the translog source. The recovery source will be the
        // original, user provided source
        if (isRecoverySourceEnabled(operation)) {
            return operation;
        }

        Pair<Function<Map<String, Object>, Map<String, Object>>> transformers = createInjectTransformer(operation);
        if (transformers == null) {
            return operation;
        }
        Function<Map<String, Object>, Map<String, Object>> injectTransformer = transformers.first();
        Function<Map<String, Object>, Map<String, Object>> maskTransformer = transformers.second();

        Tuple<? extends MediaType, Map<String, Object>> originalSource = XContentHelper.convertToMap(
            operation.parsedDoc().source(),
            true,
            operation.parsedDoc().getMediaType()
        );
        Map<String, Object> cleanVectorSource = injectTransformer.apply(originalSource.v2());
        try (BytesStreamOutput bStream = new BytesStreamOutput();) {
            XContentBuilder builder = MediaTypeRegistry.contentBuilder(originalSource.v1(), bStream).map(cleanVectorSource);
            builder.close();
            operation.parsedDoc().setSource(bStream.bytes(), XContentType.valueOf(originalSource.v1().subtype().toUpperCase(Locale.ROOT)));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // We are applying the max here and serializing to the Source's StoredField's bytes value. This is an
        // optimization. Right now, we perform the deserialization of the source in the index operation listener along
        // with the codec writer. Deserialization is very expensive. So, by applying the mask here, we avoid having to
        // do an extra large deserialization in the codec writer. At the moment, we redundantly apply the mask in the
        // writer as well as a safeguard. In the future, we can move to just masking in the listener.
        IndexableField field = operation.parsedDoc().rootDoc().getField(SourceFieldMapper.CONTENT_TYPE);
        if (field == null || field.storedValue() == null) {
            return operation;
        }
        Map<String, Object> maskedVectorSource = maskTransformer.apply(originalSource.v2());
        try (BytesStreamOutput bStream = new BytesStreamOutput();) {
            XContentBuilder builder = MediaTypeRegistry.contentBuilder(originalSource.v1(), bStream).map(maskedVectorSource);
            builder.close();
            if (field instanceof StoredField storedField) {
                storedField.setBytesValue(bStream.bytes().toBytesRef());
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return operation;
    }

    private Pair<Function<Map<String, Object>, Map<String, Object>>> createInjectTransformer(Engine.Index operation) {
        Map<String, List<Object>> injectedVectors = new HashMap<>();

        // For each document, we get the relevant vector fields to compute the injection logic
        for (ParseContext.Document document : operation.parsedDoc().docs()) {
            for (Iterator<IndexableField> it = document.iterator(); it.hasNext();) {
                IndexableField indexableField = it.next();
                if (indexableField instanceof DerivedKnnFloatVectorField knnVectorFieldType && knnVectorFieldType.isDerivedEnabled()) {
                    injectedVectors.computeIfAbsent(indexableField.name(), k -> new ArrayList<>())
                        .add(formatVector(VectorDataType.FLOAT, knnVectorFieldType.vectorValue()));
                }

                if (indexableField instanceof DerivedKnnByteVectorField knnByteVectorField && knnByteVectorField.isDerivedEnabled()) {
                    injectedVectors.computeIfAbsent(indexableField.name(), k -> new ArrayList<>())
                        .add(formatVector(VectorDataType.BYTE, knnByteVectorField.vectorValue()));
                }
            }
        }

        if (injectedVectors.isEmpty()) {
            return null;
        }

        Map<String, Function<Object, Object>> injectTransformers = new HashMap<>();
        Map<String, Function<Object, Object>> maskTransformers = new HashMap<>();
        for (Map.Entry<String, List<Object>> entry : injectedVectors.entrySet()) {
            Iterator<Object> iterator = entry.getValue().iterator();
            injectTransformers.put(entry.getKey(), (Object o) -> o == null ? o : iterator.next());
            maskTransformers.put(entry.getKey(), (Object o) -> o == null ? o : KNN10010DerivedSourceStoredFieldsWriter.MASK);
        }

        return new Pair<>(XContentMapValues.transform(injectTransformers, true), XContentMapValues.transform(maskTransformers, true));
    }

    private boolean isRecoverySourceEnabled(Engine.Index operation) {
        return operation.parsedDoc().rootDoc().getField(RECOVERY_SOURCE_NAME) != null;
    }

    protected Object formatVector(VectorDataType vectorDataType, Object vectorValue) {
        if (vectorValue instanceof byte[]) {
            BytesRef vectorBytesRef = new BytesRef((byte[]) vectorValue);
            return KNNVectorFieldMapperUtil.deserializeStoredVector(vectorBytesRef, vectorDataType);
        }
        return vectorValue;
    }

    private record Pair<T>(T first, T second) {
    }
}

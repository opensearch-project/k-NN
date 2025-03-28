/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
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
import org.opensearch.index.shard.IndexingOperationListener;
import org.opensearch.knn.index.VectorDataType;
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
        Tuple<? extends MediaType, Map<String, Object>> originalSource = XContentHelper.convertToMap(
            operation.parsedDoc().source(),
            true,
            operation.parsedDoc().getMediaType()
        );
        Map<String, Object> derivedSource = createInjectTransformer(operation).apply(originalSource.v2());

        try (BytesStreamOutput bStream = new BytesStreamOutput();) {
            XContentBuilder builder = MediaTypeRegistry.contentBuilder(originalSource.v1(), bStream).map(derivedSource);
            builder.close();
            operation.parsedDoc().setSource(bStream.bytes(), XContentType.valueOf(originalSource.v1().subtype().toUpperCase(Locale.ROOT)));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return operation;
    }

    private Function<Map<String, Object>, Map<String, Object>> createInjectTransformer(Engine.Index operation) {
        Map<String, List<Object>> injectedVectors = new HashMap<>();

        // For each document, we get the relevant vector fields to compute the injection logic
        for (ParseContext.Document document : operation.parsedDoc().docs()) {
            for (Iterator<IndexableField> it = document.iterator(); it.hasNext();) {
                IndexableField indexableField = it.next();
                if (indexableField instanceof KnnFloatVectorField knnVectorFieldType) {
                    injectedVectors.computeIfAbsent(indexableField.name(), k -> new ArrayList<>())
                        .add(formatVector(VectorDataType.FLOAT, knnVectorFieldType.vectorValue()));
                }

                if (indexableField instanceof KnnByteVectorField knnByteVectorField) {
                    injectedVectors.computeIfAbsent(indexableField.name(), k -> new ArrayList<>())
                        .add(formatVector(VectorDataType.BYTE, knnByteVectorField.vectorValue()));
                }
            }
        }
        Map<String, Function<Object, Object>> injectTransformers = new HashMap<>();
        for (Map.Entry<String, List<Object>> entry : injectedVectors.entrySet()) {
            Iterator<Object> iterator = entry.getValue().iterator();
            injectTransformers.put(entry.getKey(), (Object o) -> o == null ? o : iterator.next());
        }
        return XContentMapValues.transform(injectTransformers, true);
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
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.native_index;

import java.io.IOException;
import java.util.Map;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.FieldInfo;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.transfer.VectorTransfer;
import org.opensearch.knn.index.codec.transfer.VectorTransferByte;
import org.opensearch.knn.index.codec.transfer.VectorTransferFloat;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.codec.util.SerializationMode;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelCache;
import org.opensearch.knn.plugin.stats.KNNGraphValue;

import lombok.NonNull;

import static org.opensearch.knn.common.KNNConstants.MODEL_ID;

public abstract class KNNIndexBuilder {

    protected class NativeVectorInfo {
        protected VectorDataType vectorDataType;
        protected int dimension;
        protected SerializationMode serializationMode;
    }

    protected class NativeIndexInfo {
        protected FieldInfo fieldInfo;
        protected KNNEngine knnEngine;
        protected int numDocs;
        protected long arraySize;
        protected Map<String, Object> parameters;
        protected NativeVectorInfo vectorInfo;
        protected String indexPath;
    }

    protected final Logger logger = LogManager.getLogger(KNNIndexBuilder.class);

    public void createKNNIndex(FieldInfo fieldInfo, DocValuesProducer valuesProducer, String indexPath, boolean isMerge, boolean isRefresh) throws IOException {
        NativeIndexInfo indexInfo = getIndexInfo(fieldInfo, valuesProducer, indexPath);
        BinaryDocValues values = valuesProducer.getBinary(fieldInfo);
        if(isMerge) {
            startMergeStats(indexInfo.numDocs, indexInfo.arraySize);
        }
        if(isRefresh) {
            recordRefreshStats();
        }
        createIndex(indexInfo, values);
        if(isMerge) {
            endMergeStats(indexInfo.numDocs, indexInfo.arraySize);
        }
    }

    protected abstract void createIndex(NativeIndexInfo indexInfo, BinaryDocValues values) throws IOException;

    private long getVectorSize(NativeVectorInfo vectorInfo) {
        if(vectorInfo.vectorDataType == VectorDataType.BINARY) {
            return vectorInfo.dimension / 8;
        } else {
            return vectorInfo.dimension * 4;
        }
    }

    private NativeIndexInfo getIndexInfo(FieldInfo fieldInfo, DocValuesProducer valuesProducer, String indexPath) throws IOException {
        BinaryDocValues testValues = valuesProducer.getBinary(fieldInfo);
        NativeIndexInfo indexInfo = new NativeIndexInfo();
        indexInfo.fieldInfo = fieldInfo;
        indexInfo.knnEngine = getKNNEngine(fieldInfo);
        indexInfo.numDocs = (int)KNNCodecUtil.getTotalLiveDocsCount(testValues);
        indexInfo.vectorInfo = getVectorInfo(fieldInfo, testValues);
        indexInfo.arraySize = indexInfo.numDocs * getVectorSize(indexInfo.vectorInfo);
        indexInfo.parameters = getParameters(fieldInfo, indexInfo.knnEngine);
        indexInfo.indexPath = indexPath;
        return indexInfo;
    }

    protected abstract Map<String, Object> getParameters(FieldInfo fieldInfo, KNNEngine knnEngine) throws IOException;

    protected abstract NativeVectorInfo getVectorInfo(FieldInfo fieldInfo, BinaryDocValues testValues) throws IOException;

    protected VectorTransfer getVectorTransfer(VectorDataType vectorDataType) {
        if (VectorDataType.BINARY == vectorDataType) {
            return new VectorTransferByte(KNNSettings.getVectorStreamingMemoryLimit().getBytes());
        }
        return new VectorTransferFloat(KNNSettings.getVectorStreamingMemoryLimit().getBytes());
    }

    protected KNNEngine getKNNEngine(@NonNull FieldInfo field) {
        final String modelId = field.attributes().get(MODEL_ID);
        if (modelId != null) {
            var model = ModelCache.getInstance().get(modelId);
            return model.getModelMetadata().getKnnEngine();
        }
        final String engineName = field.attributes().getOrDefault(KNNConstants.KNN_ENGINE, KNNEngine.DEFAULT.getName());
        return KNNEngine.getEngine(engineName);
    }

    private void startMergeStats(int numDocs, long arraySize) {
        KNNGraphValue.MERGE_CURRENT_OPERATIONS.increment();
        KNNGraphValue.MERGE_CURRENT_DOCS.incrementBy(numDocs);
        KNNGraphValue.MERGE_CURRENT_SIZE_IN_BYTES.incrementBy(arraySize);
        KNNGraphValue.MERGE_TOTAL_OPERATIONS.increment();
        KNNGraphValue.MERGE_TOTAL_DOCS.incrementBy(numDocs);
        KNNGraphValue.MERGE_TOTAL_SIZE_IN_BYTES.incrementBy(arraySize);
    }

    private void endMergeStats(int numDocs, long arraySize) {
        KNNGraphValue.MERGE_CURRENT_OPERATIONS.decrement();
        KNNGraphValue.MERGE_CURRENT_DOCS.decrementBy(numDocs);
        KNNGraphValue.MERGE_CURRENT_SIZE_IN_BYTES.decrementBy(arraySize);
    }

    private void recordRefreshStats() {
        KNNGraphValue.REFRESH_TOTAL_OPERATIONS.increment();
    }
}

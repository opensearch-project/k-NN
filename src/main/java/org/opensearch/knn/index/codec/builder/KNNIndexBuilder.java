/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.builder;

import java.io.IOException;
import java.util.Map;

import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.transfer.VectorTransfer;
import org.opensearch.knn.index.codec.transfer.VectorTransferByte;
import org.opensearch.knn.index.codec.transfer.VectorTransferFloat;
import org.opensearch.knn.index.codec.util.SerializationMode;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelCache;
import org.opensearch.knn.plugin.stats.KNNGraphValue;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

import static org.opensearch.knn.common.KNNConstants.MODEL_ID;

public abstract class KNNIndexBuilder {
    public static String FROM_SCRATCH_ITERATIVE = "FROM_SCRATCH_ITERATIVE";
    public static String FROM_SCRATCH = "FROM_SCRATCH";
    public static String FROM_TEMPLATE = "FROM_TEMPLATE";
    @Getter
    @Setter
    protected boolean isMerge;
    @Getter
    @Setter
    protected boolean isRefresh;
    @Getter
    @Setter
    protected FieldInfo fieldInfo;
    @Getter
    @Setter
    protected DocValuesProducer valuesProducer;
    @Getter
    @Setter
    protected String indexPath;

    protected KNNEngine knnEngine;
    protected long numDocs;
    protected VectorDataType vectorDataType;
    protected VectorTransfer vectorTransfer;
    protected SerializationMode serializationMode;
    protected Map<String, Object> parameters;
    protected long arraySize;
    protected BinaryDocValues values;
    protected int dimension;

    public void createKNNIndex() throws IOException {
        getInfoFromField();
        genParameters();
        genDatasetMetrics();
        genVectorMetrics();
        recordRefreshStats();
        startMergeStats();
        createIndex();
        endMergeStats();
    }

    protected abstract void createIndex() throws IOException;

    protected abstract void genParameters() throws IOException;

    protected abstract void genDatasetMetrics() throws IOException;

    protected void getInfoFromField() throws IOException {
        knnEngine = getKNNEngine(fieldInfo);
        values = valuesProducer.getBinary(fieldInfo);
    }

    protected void genVectorMetrics() throws IOException {
        BinaryDocValues testValues = valuesProducer.getBinary(fieldInfo);
        // Hack to get the data metrics from the first document. We account for this in KNNCodecUtil.
        testValues.nextDoc();
        BytesRef firstDoc = testValues.binaryValue();
        vectorTransfer = getVectorTransfer(vectorDataType);
        serializationMode = vectorTransfer.getSerializationMode(firstDoc);
        arraySize = numDocs * firstDoc.length;
        if (vectorDataType == VectorDataType.BINARY) {
            dimension = firstDoc.length * 8;
        } else {
            dimension = firstDoc.length / 4;
        }
    }

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

    private void startMergeStats() {
        if (!isMerge) {
            return;
        }
        KNNGraphValue.MERGE_CURRENT_OPERATIONS.increment();
        KNNGraphValue.MERGE_CURRENT_DOCS.incrementBy(numDocs);
        KNNGraphValue.MERGE_CURRENT_SIZE_IN_BYTES.incrementBy(arraySize);
        KNNGraphValue.MERGE_TOTAL_OPERATIONS.increment();
        KNNGraphValue.MERGE_TOTAL_DOCS.incrementBy(numDocs);
        KNNGraphValue.MERGE_TOTAL_SIZE_IN_BYTES.incrementBy(arraySize);
    }

    private void endMergeStats() {
        if (!isMerge) {
            return;
        }
        KNNGraphValue.MERGE_CURRENT_OPERATIONS.decrement();
        KNNGraphValue.MERGE_CURRENT_DOCS.decrementBy(numDocs);
        KNNGraphValue.MERGE_CURRENT_SIZE_IN_BYTES.decrementBy(arraySize);
    }

    private void recordRefreshStats() {
        if (!isRefresh) {
            return;
        }
        KNNGraphValue.REFRESH_TOTAL_OPERATIONS.increment();
    }
}

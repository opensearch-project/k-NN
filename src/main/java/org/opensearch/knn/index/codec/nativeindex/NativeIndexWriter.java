/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

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

import lombok.Builder;
import lombok.NonNull;
import lombok.Value;

import static org.opensearch.knn.common.KNNConstants.MODEL_ID;

/**
 * Abstract class to build the KNN index and write it to disk
 */
public abstract class NativeIndexWriter {

    /**
     * Class that holds info about vectors
     */
    @Builder
    @Value
    protected static class NativeVectorInfo {
        private VectorDataType vectorDataType;
        private int dimension;
        private SerializationMode serializationMode;
    }

    /**
     * Class that holds info about the native index
     */
    @Builder
    @Value
    protected static class NativeIndexInfo {
        private FieldInfo fieldInfo;
        private KNNEngine knnEngine;
        private int numDocs;
        private long arraySize;
        private Map<String, Object> parameters;
        private NativeVectorInfo vectorInfo;
        private String indexPath;
    }

    protected final Logger logger = LogManager.getLogger(NativeIndexWriter.class);

    /**
     * Method for creating a KNN index in the specified native library
     *
     * @param fieldInfo
     * @param valuesProducer
     * @param indexPath
     * @param isMerge
     * @param isRefresh
     * @throws IOException
     */
    public void createKNNIndex(FieldInfo fieldInfo, DocValuesProducer valuesProducer, String indexPath, boolean isMerge, boolean isRefresh)
        throws IOException {
        NativeIndexInfo indexInfo = getIndexInfo(fieldInfo, valuesProducer, indexPath);
        BinaryDocValues values = valuesProducer.getBinary(fieldInfo);
        if (isMerge) {
            startMergeStats(indexInfo.numDocs, indexInfo.arraySize);
        }
        if (isRefresh) {
            recordRefreshStats();
        }
        createIndex(indexInfo, values);
        if (isMerge) {
            endMergeStats(indexInfo.numDocs, indexInfo.arraySize);
        }
    }

    /**
     * Method that makes a native index given the parameters from indexInfo
     * @param indexInfo
     * @param values
     * @throws IOException
     */
    protected abstract void createIndex(NativeIndexInfo indexInfo, BinaryDocValues values) throws IOException;

    /**
     * Method that generates extra index parameters to be passed to the native library
     * @param fieldInfo
     * @param knnEngine
     * @return extra index parameters to be passed to the native library
     * @throws IOException
     */
    protected abstract Map<String, Object> getParameters(FieldInfo fieldInfo, KNNEngine knnEngine) throws IOException;

    /**
     * Method that gets the native vector info
     * @param fieldInfo
     * @param testValues
     * @return native vector info
     * @throws IOException
     */
    protected abstract NativeVectorInfo getVectorInfo(FieldInfo fieldInfo, DocValuesProducer valuesProducer) throws IOException;

    protected VectorTransfer getVectorTransfer(VectorDataType vectorDataType) {
        if (VectorDataType.BINARY == vectorDataType) {
            return new VectorTransferByte(KNNSettings.getVectorStreamingMemoryLimit().getBytes());
        }
        return new VectorTransferFloat(KNNSettings.getVectorStreamingMemoryLimit().getBytes());
    }

    /**
     * Method that gets the native index info from a given field
     * @param fieldInfo
     * @param valuesProducer
     * @param indexPath
     * @return native index info
     * @throws IOException
     */
    private NativeIndexInfo getIndexInfo(FieldInfo fieldInfo, DocValuesProducer valuesProducer, String indexPath) throws IOException {
        int numDocs = (int) KNNCodecUtil.getTotalLiveDocsCount(valuesProducer.getBinary(fieldInfo));
        NativeVectorInfo vectorInfo = getVectorInfo(fieldInfo, valuesProducer);
        KNNEngine knnEngine = getKNNEngine(fieldInfo);
        NativeIndexInfo indexInfo = NativeIndexInfo.builder()
            .fieldInfo(fieldInfo)
            .knnEngine(getKNNEngine(fieldInfo))
            .numDocs((int) numDocs)
            .vectorInfo(vectorInfo)
            .arraySize(numDocs * getBytesPerVector(vectorInfo))
            .parameters(getParameters(fieldInfo, knnEngine))
            .indexPath(indexPath)
            .build();
        return indexInfo;
    }

    private long getBytesPerVector(NativeVectorInfo vectorInfo) {
        if (vectorInfo.vectorDataType == VectorDataType.BINARY) {
            return vectorInfo.dimension / 8;
        } else {
            return vectorInfo.dimension * 4;
        }
    }

    private KNNEngine getKNNEngine(@NonNull FieldInfo field) {
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

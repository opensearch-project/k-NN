/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.builder;

import java.io.IOException;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.HashMap;
import java.util.Map;

import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.FieldInfo;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.IndexUtil;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.transfer.VectorTransfer;
import org.opensearch.knn.index.codec.transfer.VectorTransferByte;
import org.opensearch.knn.index.codec.transfer.VectorTransferFloat;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.codec.util.SerializationMode;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelCache;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.plugin.stats.KNNGraphValue;

import lombok.Getter;
import lombok.Setter;

import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.index.util.Faiss.FAISS_BINARY_INDEX_DESCRIPTION_PREFIX;

public class KNNIndexBuilder {
    protected boolean isMerge;
    protected boolean isRefresh;
    @Getter
    @Setter
    protected FieldInfo fieldInfo;
    @Getter
    @Setter
    protected BinaryDocValues values;
    @Getter
    @Setter
    protected KNNEngine knnEngine;
    @Getter
    @Setter
    protected String indexPath;
    @Getter
    @Setter
    protected boolean fromScratch;
    @Getter
    @Setter
    protected boolean iterative;
    protected long numDocs;
    protected VectorDataType vectorDataType;
    protected VectorTransfer vectorTransfer;
    protected SerializationMode serializationMode;
    protected Map<String, Object> parameters;

    public KNNIndexBuilder() {
        this.isMerge = false;
        this.isRefresh = false;
        this.fieldInfo = null;
        this.values = null;
        this.knnEngine = null;
        this.indexPath = null;
        this.fromScratch = false;
        this.iterative = false;
    }

    public void createKNNIndex() throws IOException {
        Map<String, Object> parameters = genParameters(fromScratch, fieldInfo, knnEngine);
        if (fromScratch && iterative) {
            createKNNIndexFromScratchIteratively(fieldInfo, values, knnEngine, indexPath, parameters);
        } else if (fromScratch) {
            createKNNIndexFromScratch(fieldInfo, values, knnEngine, indexPath, parameters);
        } else {
            createKNNIndexFromTemplate(fieldInfo, values, knnEngine, indexPath, parameters);
        }
    }

    public void doMerge() {
        this.isMerge = true;
        KNNGraphValue.MERGE_CURRENT_OPERATIONS.increment();
    }

    public void doRefresh() {
        this.isRefresh = true;
        recordRefreshStats();
    }

    private void currentMergeStats(int length, long arraySize) {
        KNNGraphValue.MERGE_CURRENT_OPERATIONS.increment();
        KNNGraphValue.MERGE_CURRENT_DOCS.incrementBy(length);
        KNNGraphValue.MERGE_CURRENT_SIZE_IN_BYTES.incrementBy(arraySize);
        KNNGraphValue.MERGE_TOTAL_OPERATIONS.increment();
        KNNGraphValue.MERGE_TOTAL_DOCS.incrementBy(length);
        KNNGraphValue.MERGE_TOTAL_SIZE_IN_BYTES.incrementBy(arraySize);
    }

    private void recordMergeStats(int length, long arraySize) {
        KNNGraphValue.MERGE_CURRENT_OPERATIONS.decrement();
        KNNGraphValue.MERGE_CURRENT_DOCS.decrementBy(length);
        KNNGraphValue.MERGE_CURRENT_SIZE_IN_BYTES.decrementBy(arraySize);
    }

    private void recordRefreshStats() {
        KNNGraphValue.REFRESH_TOTAL_OPERATIONS.increment();
    }

    private long initIndexFromScratch(long size, int dim, KNNEngine knnEngine, Map<String, Object> parameters) throws IOException {
        // Pass the path for the nms library to save the file
        return AccessController.doPrivileged((PrivilegedAction<Long>) () -> {
            return JNIService.initIndexFromScratch(size, dim, parameters, knnEngine);
        });
    }

    private void insertToIndex(KNNCodecUtil.VectorBatch pair, KNNEngine knnEngine, long indexAddress, Map<String, Object> parameters)
        throws IOException {
        // Could be zero docs because of edge cases with batch creation
        if (pair.docs.length == 0) return;
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            JNIService.insertToIndex(pair.docs, pair.getVectorAddress(), pair.getDimension(), parameters, indexAddress, knnEngine);
            return null;
        });
    }

    private void writeIndex(long indexAddress, String indexPath, KNNEngine knnEngine, Map<String, Object> parameters) throws IOException {
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            JNIService.writeIndex(indexPath, indexAddress, knnEngine, parameters);
            return null;
        });
    }

    private Map<String, Object> genParameters(boolean fromScratch, FieldInfo fieldInfo, KNNEngine knnEngine) throws IOException {
        Map<String, Object> parameters = new HashMap<>();
        if (fromScratch) {
            Map<String, String> fieldAttributes = fieldInfo.attributes();
            String parametersString = fieldAttributes.get(KNNConstants.PARAMETERS);

            // parametersString will be null when legacy mapper is used
            if (parametersString == null) {
                parameters.put(
                    KNNConstants.SPACE_TYPE,
                    fieldAttributes.getOrDefault(KNNConstants.SPACE_TYPE, SpaceType.DEFAULT.getValue())
                );

                String efConstruction = fieldAttributes.get(KNNConstants.HNSW_ALGO_EF_CONSTRUCTION);
                Map<String, Object> algoParams = new HashMap<>();
                if (efConstruction != null) {
                    algoParams.put(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, Integer.parseInt(efConstruction));
                }

                String m = fieldAttributes.get(KNNConstants.HNSW_ALGO_M);
                if (m != null) {
                    algoParams.put(KNNConstants.METHOD_PARAMETER_M, Integer.parseInt(m));
                }
                parameters.put(PARAMETERS, algoParams);
            } else {
                parameters.putAll(
                    XContentHelper.createParser(
                        NamedXContentRegistry.EMPTY,
                        DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                        new BytesArray(parametersString),
                        MediaTypeRegistry.getDefaultMediaType()
                    ).map()
                );
            }

            // Update index description of Faiss for binary data type
            if (KNNEngine.FAISS == knnEngine
                && VectorDataType.BINARY.getValue()
                    .equals(fieldAttributes.getOrDefault(KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.DEFAULT.getValue()))
                && parameters.get(KNNConstants.INDEX_DESCRIPTION_PARAMETER) != null) {
                parameters.put(
                    KNNConstants.INDEX_DESCRIPTION_PARAMETER,
                    FAISS_BINARY_INDEX_DESCRIPTION_PREFIX + parameters.get(KNNConstants.INDEX_DESCRIPTION_PARAMETER).toString()
                );
                IndexUtil.updateVectorDataTypeToParameters(parameters, VectorDataType.BINARY);
            }
        }
        // Used to determine how many threads to use when indexing
        parameters.put(KNNConstants.INDEX_THREAD_QTY, KNNSettings.state().getSettingValue(KNNSettings.KNN_ALGO_PARAM_INDEX_THREAD_QTY));
        return parameters;
    }

    private void createKNNIndexFromTemplate(
        FieldInfo field,
        BinaryDocValues values,
        KNNEngine knnEngine,
        String indexPath,
        Map<String, Object> parameters
    ) throws IOException {
        String modelId = field.attributes().get(MODEL_ID);
        Model model = ModelCache.getInstance().get(modelId);
        if (model.getModelBlob() == null) {
            throw new RuntimeException(String.format("There is no trained model with id \"%s\"", modelId));
        }
        byte[] modelBlob = model.getModelBlob();
        IndexUtil.updateVectorDataTypeToParameters(parameters, model.getModelMetadata().getVectorDataType());
        VectorDataType vectorDataType = model.getModelMetadata().getVectorDataType();
        KNNCodecUtil.VectorBatch batch = KNNCodecUtil.getVectorBatch(values, getVectorTransfer(vectorDataType), false);

        int numDocs = (int) KNNCodecUtil.getTotalLiveDocsCount(values);

        if (numDocs == 0) {
            return;
        }

        long arraySize = KNNCodecUtil.calculateArraySize(numDocs, batch.getDimension(), batch.serializationMode);

        if (isMerge) {
            currentMergeStats(numDocs, arraySize);
        }

        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            JNIService.createIndexFromTemplate(
                batch.docs,
                batch.getVectorAddress(),
                batch.getDimension(),
                indexPath,
                modelBlob,
                parameters,
                knnEngine
            );
            return null;
        });

        if (isMerge) {
            recordMergeStats(numDocs, arraySize);
        }
    }

    private void createKNNIndexFromScratch(
        FieldInfo fieldInfo,
        BinaryDocValues values,
        KNNEngine knnEngine,
        String indexPath,
        Map<String, Object> parameters
    ) throws IOException {
        Map<String, String> fieldAttributes = fieldInfo.attributes();
        VectorDataType vectorDataType = VectorDataType.get(
            fieldAttributes.getOrDefault(KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.DEFAULT.getValue())
        );
        VectorTransfer transfer = getVectorTransfer(vectorDataType);
        KNNCodecUtil.VectorBatch batch = KNNCodecUtil.getVectorBatch(values, transfer, false);

        int numDocs = (int) KNNCodecUtil.getTotalLiveDocsCount(values);

        if (numDocs == 0) {
            return;
        }

        long arraySize = KNNCodecUtil.calculateArraySize(numDocs, batch.getDimension(), batch.serializationMode);

        if (isMerge) {
            currentMergeStats(numDocs, arraySize);
        }

        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            JNIService.createIndex(batch.docs, batch.getVectorAddress(), batch.getDimension(), indexPath, parameters, knnEngine);
            return null;
        });

        if (isMerge) {
            recordMergeStats(numDocs, arraySize);
        }
    }

    private void createKNNIndexFromScratchIteratively(
        FieldInfo fieldInfo,
        BinaryDocValues values,
        KNNEngine knnEngine,
        String indexPath,
        Map<String, Object> parameters
    ) throws IOException {
        Map<String, String> fieldAttributes = fieldInfo.attributes();
        VectorDataType vectorDataType = VectorDataType.get(
            fieldAttributes.getOrDefault(KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.DEFAULT.getValue())
        );
        VectorTransfer transfer = getVectorTransfer(vectorDataType);
        KNNCodecUtil.VectorBatch batch = KNNCodecUtil.getVectorBatch(values, transfer, true);

        int numDocs = (int) KNNCodecUtil.getTotalLiveDocsCount(values);

        if (numDocs == 0) {
            return;
        }

        long arraySize = KNNCodecUtil.calculateArraySize(numDocs, batch.getDimension(), batch.serializationMode);

        if (isMerge) {
            currentMergeStats(numDocs, arraySize);
        }

        long indexAddress = initIndexFromScratch(numDocs, batch.getDimension(), knnEngine, parameters);
        for (; !batch.finished; batch = KNNCodecUtil.getVectorBatch(values, transfer, true)) {
            insertToIndex(batch, knnEngine, indexAddress, parameters);
        }
        insertToIndex(batch, knnEngine, indexAddress, parameters);
        writeIndex(indexAddress, indexPath, knnEngine, parameters);
        if (isMerge) {
            recordMergeStats(numDocs, arraySize);
        }
    }

    private VectorTransfer getVectorTransfer(VectorDataType vectorDataType) {
        if (VectorDataType.BINARY == vectorDataType) {
            return new VectorTransferByte(KNNSettings.getVectorStreamingMemoryLimit().getBytes());
        }
        return new VectorTransferFloat(KNNSettings.getVectorStreamingMemoryLimit().getBytes());
    }
}

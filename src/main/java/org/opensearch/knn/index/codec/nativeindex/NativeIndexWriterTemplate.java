/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import java.io.IOException;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.HashMap;
import java.util.Map;

import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.IndexUtil;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.transfer.VectorTransfer;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelCache;
import org.opensearch.knn.jni.JNIService;

import static org.opensearch.knn.common.KNNConstants.MODEL_ID;

/**
 * Abstract class to build the KNN index from a template model and write it to disk
 */
public class NativeIndexWriterTemplate extends NativeIndexWriter {

    protected void createIndex(NativeIndexInfo indexInfo, BinaryDocValues values) throws IOException {
        String modelId = indexInfo.getFieldInfo().attributes().get(MODEL_ID);
        Model model = ModelCache.getInstance().get(modelId);
        if (model.getModelBlob() == null) {
            throw new RuntimeException(String.format("There is no trained model with id \"%s\"", modelId));
        }
        byte[] modelBlob = model.getModelBlob();
        IndexUtil.updateVectorDataTypeToParameters(indexInfo.getParameters(), model.getModelMetadata().getVectorDataType());
        // This is carried over from the old index creation process. Why can't we get the vector data type
        // by just reading it from the field?
        VectorDataType vectorDataType = model.getModelMetadata().getVectorDataType();
        KNNCodecUtil.VectorBatch batch = KNNCodecUtil.getVectorBatch(values, getVectorTransfer(vectorDataType), false);

        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            JNIService.createIndexFromTemplate(
                batch.docs,
                batch.getVectorAddress(),
                batch.getDimension(),
                indexInfo.getIndexPath(),
                modelBlob,
                indexInfo.getParameters(),
                indexInfo.getKnnEngine()
            );
            return null;
        });
    }

    @Override
    protected Map<String, Object> getParameters(FieldInfo fieldInfo, KNNEngine knnEngine) throws IOException {
        Map<String, Object> parameters = new HashMap<>();
        parameters.put(KNNConstants.INDEX_THREAD_QTY, KNNSettings.state().getSettingValue(KNNSettings.KNN_ALGO_PARAM_INDEX_THREAD_QTY));
        String modelId = fieldInfo.attributes().get(MODEL_ID);
        Model model = ModelCache.getInstance().get(modelId);
        if (model.getModelBlob() == null) {
            throw new RuntimeException(String.format("There is no trained model with id \"%s\"", modelId));
        }
        IndexUtil.updateVectorDataTypeToParameters(parameters, model.getModelMetadata().getVectorDataType());
        return parameters;
    }

    @Override
    protected NativeVectorInfo getVectorInfo(FieldInfo fieldInfo, BinaryDocValues testValues) throws IOException {
        testValues.nextDoc();
        BytesRef firstDoc = testValues.binaryValue();
        String modelId = fieldInfo.attributes().get(MODEL_ID);
        Model model = ModelCache.getInstance().get(modelId);
        if (model.getModelBlob() == null) {
            throw new RuntimeException(String.format("There is no trained model with id \"%s\"", modelId));
        }
        VectorDataType vectorDataType = model.getModelMetadata().getVectorDataType();
        VectorTransfer vectorTransfer = getVectorTransfer(vectorDataType);
        int dimension = 0;
        if (vectorDataType == VectorDataType.BINARY) {
            dimension = firstDoc.length * 8;
        } else {
            dimension = firstDoc.length / 4;
        }
        NativeVectorInfo vectorInfo = NativeVectorInfo.builder()
        .vectorDataType(vectorDataType)
        .dimension(dimension)
        .serializationMode(vectorTransfer.getSerializationMode(firstDoc))
        .build();
        return vectorInfo;
    }
}

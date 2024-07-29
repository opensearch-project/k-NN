/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.native_index;

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

public class NativeIndexBuilderTemplate extends NativeIndexBuilder {

    protected void createIndex(NativeIndexInfo indexInfo, BinaryDocValues values) throws IOException {
        String modelId = indexInfo.fieldInfo.attributes().get(MODEL_ID);
        Model model = ModelCache.getInstance().get(modelId);
        if (model.getModelBlob() == null) {
            throw new RuntimeException(String.format("There is no trained model with id \"%s\"", modelId));
        }
        byte[] modelBlob = model.getModelBlob();
        IndexUtil.updateVectorDataTypeToParameters(indexInfo.parameters, model.getModelMetadata().getVectorDataType());
        indexInfo.vectorInfo.vectorDataType = model.getModelMetadata().getVectorDataType();
        KNNCodecUtil.VectorBatch batch = KNNCodecUtil.getVectorBatch(values, getVectorTransfer(indexInfo.vectorInfo.vectorDataType), false);

        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            JNIService.createIndexFromTemplate(
                batch.docs,
                batch.getVectorAddress(),
                batch.getDimension(),
                indexInfo.indexPath,
                modelBlob,
                indexInfo.parameters,
                indexInfo.knnEngine
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
        NativeVectorInfo vectorInfo = new NativeVectorInfo();
        testValues.nextDoc();
        BytesRef firstDoc = testValues.binaryValue();
        String modelId = fieldInfo.attributes().get(MODEL_ID);
        Model model = ModelCache.getInstance().get(modelId);
        if (model.getModelBlob() == null) {
            throw new RuntimeException(String.format("There is no trained model with id \"%s\"", modelId));
        }
        vectorInfo.vectorDataType = model.getModelMetadata().getVectorDataType();
        VectorTransfer vectorTransfer = getVectorTransfer(vectorInfo.vectorDataType);
        vectorInfo.serializationMode = vectorTransfer.getSerializationMode(firstDoc);
        if (vectorInfo.vectorDataType == VectorDataType.BINARY) {
            vectorInfo.dimension = firstDoc.length * 8;
        } else {
            vectorInfo.dimension = firstDoc.length / 4;
        }
        return vectorInfo;
    }
}

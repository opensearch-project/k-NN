/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.builder;

import java.io.IOException;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.HashMap;

import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.IndexUtil;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelCache;
import org.opensearch.knn.jni.JNIService;

import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.getTotalLiveDocsCount;

public class KNNIndexBuilderTemplate extends KNNIndexBuilder {
    protected byte[] modelBlob;

    protected void getInfoFromField() throws IOException {
        knnEngine = getKNNEngine(fieldInfo);
        values = valuesProducer.getBinary(fieldInfo);
    }

    protected void genParameters() throws IOException {
        parameters = new HashMap<>();
        parameters.put(KNNConstants.INDEX_THREAD_QTY, KNNSettings.state().getSettingValue(KNNSettings.KNN_ALGO_PARAM_INDEX_THREAD_QTY));
    }

    protected void genDatasetMetrics() throws IOException {
        numDocs = getTotalLiveDocsCount(values);
        String modelId = fieldInfo.attributes().get(MODEL_ID);
        Model model = ModelCache.getInstance().get(modelId);
        if (model.getModelBlob() == null) {
            throw new RuntimeException(String.format("There is no trained model with id \"%s\"", modelId));
        }
        modelBlob = model.getModelBlob();
        IndexUtil.updateVectorDataTypeToParameters(parameters, model.getModelMetadata().getVectorDataType());
        vectorDataType = model.getModelMetadata().getVectorDataType();
    }

    protected void createIndex() throws IOException {
        KNNCodecUtil.VectorBatch batch = KNNCodecUtil.getVectorBatch(values, getVectorTransfer(vectorDataType), false);

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
    }
}

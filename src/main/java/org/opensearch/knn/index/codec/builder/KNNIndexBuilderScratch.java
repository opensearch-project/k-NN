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
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.jni.JNIService;

import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.getTotalLiveDocsCount;
import static org.opensearch.knn.index.util.Faiss.FAISS_BINARY_INDEX_DESCRIPTION_PREFIX;

public class KNNIndexBuilderScratch extends KNNIndexBuilder {

    protected void getInfoFromField() throws IOException {
        knnEngine = getKNNEngine(fieldInfo);
        values = valuesProducer.getBinary(fieldInfo);
    }

    protected void genParameters() throws IOException {
        parameters = new HashMap<>();
        Map<String, String> fieldAttributes = fieldInfo.attributes();
        String parametersString = fieldAttributes.get(KNNConstants.PARAMETERS);

        // parametersString will be null when legacy mapper is used
        if (parametersString == null) {
            parameters.put(KNNConstants.SPACE_TYPE, fieldAttributes.getOrDefault(KNNConstants.SPACE_TYPE, SpaceType.DEFAULT.getValue()));

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
        // Used to determine how many threads to use when indexing
        parameters.put(KNNConstants.INDEX_THREAD_QTY, KNNSettings.state().getSettingValue(KNNSettings.KNN_ALGO_PARAM_INDEX_THREAD_QTY));
    }

    protected void genDatasetMetrics() throws IOException {
        numDocs = getTotalLiveDocsCount(values);
        vectorDataType = VectorDataType.get(
            fieldInfo.attributes().getOrDefault(KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.DEFAULT.getValue())
        );
    }

    protected void createIndex() throws IOException {
        KNNCodecUtil.VectorBatch batch = KNNCodecUtil.getVectorBatch(values, vectorTransfer, false);
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            JNIService.createIndex(batch.docs, batch.getVectorAddress(), batch.getDimension(), indexPath, parameters, knnEngine);
            return null;
        });
    }
}

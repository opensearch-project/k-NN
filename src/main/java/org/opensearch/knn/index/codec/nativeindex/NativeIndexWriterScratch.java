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

import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.util.BytesRef;
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
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.jni.JNIService;

import lombok.extern.log4j.Log4j2;

import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.index.util.Faiss.FAISS_BINARY_INDEX_DESCRIPTION_PREFIX;

/**
 * Class to build the KNN index from scratch and write it to disk
 */
@Log4j2
public class NativeIndexWriterScratch extends NativeIndexWriter {

    protected NativeVectorInfo getVectorInfo(FieldInfo fieldInfo, DocValuesProducer valuesProducer) throws IOException {
        // Hack to get the data metrics from the first document. We account for this in KNNCodecUtil.
        BinaryDocValues testValues = valuesProducer.getBinary(fieldInfo);
        testValues.nextDoc();
        BytesRef firstDoc = testValues.binaryValue();
        VectorDataType vectorDataType = VectorDataType.get(
            fieldInfo.attributes().getOrDefault(KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.DEFAULT.getValue())
        );
        int dimension = 0;
        if (vectorDataType == VectorDataType.BINARY) {
            dimension = firstDoc.length * 8;
        } else {
            dimension = firstDoc.length / 4;
        }
        NativeVectorInfo vectorInfo = NativeVectorInfo.builder().vectorDataType(vectorDataType).dimension(dimension).build();
        return vectorInfo;
    }

    protected Map<String, Object> getParameters(FieldInfo fieldInfo, KNNEngine knnEngine) throws IOException {
        Map<String, Object> parameters = new HashMap<>();
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
        return parameters;
    }

    protected void createIndex(NativeIndexInfo indexInfo, BinaryDocValues values) throws IOException {
        VectorTransfer vectorTransfer = getVectorTransfer(indexInfo.getVectorInfo().getVectorDataType());
        KNNCodecUtil.VectorBatch batch = KNNCodecUtil.getVectorBatch(values, vectorTransfer, false);
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            JNIService.createIndex(
                batch.docs,
                batch.getVectorAddress(),
                batch.getDimension(),
                indexInfo.getIndexPath(),
                indexInfo.getParameters(),
                indexInfo.getKnnEngine()
            );
            return null;
        });
    }
}

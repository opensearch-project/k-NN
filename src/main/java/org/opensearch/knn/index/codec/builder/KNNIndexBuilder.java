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
import lombok.NonNull;
import lombok.Setter;

import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.getTotalLiveDocsCount;
import static org.opensearch.knn.index.util.Faiss.FAISS_BINARY_INDEX_DESCRIPTION_PREFIX;

public class KNNIndexBuilder {
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
    protected boolean fromScratch;
    protected boolean iterative;
    protected String creationMethod;
    protected long numDocs;
    protected VectorDataType vectorDataType;
    protected VectorTransfer vectorTransfer;
    protected SerializationMode serializationMode;
    protected Map<String, Object> parameters;
    protected byte[] modelBlob;
    protected long arraySize;
    protected BinaryDocValues values;

    @FunctionalInterface
    private interface NativeIndexCreator {
        void createIndex() throws IOException;
    }

    protected final Map<String, NativeIndexCreator> methods;

    public KNNIndexBuilder() {
        methods = new HashMap<>();
        // Add all methods here
        methods.put(FROM_SCRATCH_ITERATIVE, () -> { createKNNIndexFromScratchIteratively(); });
        methods.put(FROM_SCRATCH, () -> { createKNNIndexFromScratch(); });
        methods.put(FROM_TEMPLATE, () -> { createKNNIndexFromTemplate(); });
    }

    public void createKNNIndex() throws IOException {
        getInfoFromField();
        genParameters();
        genDatasetMetrics();
        recordRefreshStats();
        startMergeStats();
        methods.get(creationMethod).createIndex();
        endMergeStats();
    }

    private void getInfoFromField() throws IOException {
        fromScratch = !fieldInfo.attributes().containsKey(MODEL_ID);
        knnEngine = getKNNEngine(fieldInfo);
        iterative = fromScratch && KNNEngine.FAISS == knnEngine;
        if (fromScratch && iterative) {
            creationMethod = KNNIndexBuilder.FROM_SCRATCH_ITERATIVE;
        } else if (fromScratch) {
            creationMethod = KNNIndexBuilder.FROM_SCRATCH;
        } else {
            creationMethod = KNNIndexBuilder.FROM_TEMPLATE;
        }
        values = valuesProducer.getBinary(fieldInfo);
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

    private void genParameters() throws IOException {
        parameters = new HashMap<>();
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
    }

    private void genDatasetMetrics() throws IOException {
        numDocs = getTotalLiveDocsCount(values);
        if (fromScratch) {
            vectorDataType = VectorDataType.get(
                fieldInfo.attributes().getOrDefault(KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.DEFAULT.getValue())
            );
        } else {
            String modelId = fieldInfo.attributes().get(MODEL_ID);
            Model model = ModelCache.getInstance().get(modelId);
            if (model.getModelBlob() == null) {
                throw new RuntimeException(String.format("There is no trained model with id \"%s\"", modelId));
            }
            modelBlob = model.getModelBlob();
            IndexUtil.updateVectorDataTypeToParameters(parameters, model.getModelMetadata().getVectorDataType());
            vectorDataType = model.getModelMetadata().getVectorDataType();
        }
        BinaryDocValues testValues = valuesProducer.getBinary(fieldInfo);
        // Hack to get the data metrics from the first document. We account for this in KNNCodecUtil.
        testValues.nextDoc();
        BytesRef firstDoc = testValues.binaryValue();
        vectorTransfer = getVectorTransfer(vectorDataType);
        serializationMode = vectorTransfer.getSerializationMode(firstDoc);
        arraySize = numDocs * firstDoc.length;
    }

    private void createKNNIndexFromTemplate() throws IOException {
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

    private void createKNNIndexFromScratch() throws IOException {
        KNNCodecUtil.VectorBatch batch = KNNCodecUtil.getVectorBatch(values, vectorTransfer, false);
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            JNIService.createIndex(batch.docs, batch.getVectorAddress(), batch.getDimension(), indexPath, parameters, knnEngine);
            return null;
        });
    }

    private void createKNNIndexFromScratchIteratively() throws IOException {
        KNNCodecUtil.VectorBatch batch = KNNCodecUtil.getVectorBatch(values, vectorTransfer, true);
        long indexAddress = initIndexFromScratch(numDocs, batch.getDimension(), knnEngine, parameters);
        try {
            while(true) {
                insertToIndex(batch, knnEngine, indexAddress, parameters);
                if(batch.finished) {
                    break;
                }
                batch = KNNCodecUtil.getVectorBatch(values, vectorTransfer, true);
            }
            writeIndex(indexAddress, indexPath, knnEngine, parameters);
        } catch (Exception e) {
            JNIService.free(indexAddress, knnEngine, VectorDataType.BINARY == vectorDataType);
            throw e;
        }
    }

    private VectorTransfer getVectorTransfer(VectorDataType vectorDataType) {
        if (VectorDataType.BINARY == vectorDataType) {
            return new VectorTransferByte(KNNSettings.getVectorStreamingMemoryLimit().getBytes());
        }
        return new VectorTransferFloat(KNNSettings.getVectorStreamingMemoryLimit().getBytes());
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
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN80Codec;

import lombok.NonNull;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.ChecksumIndexInput;
import org.opensearch.common.StopWatch;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.knn.index.IndexUtil;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.transfer.VectorTransfer;
import org.opensearch.knn.index.codec.transfer.VectorTransferByte;
import org.opensearch.knn.index.codec.transfer.VectorTransferFloat;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelCache;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.codecs.DocValuesConsumer;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.plugin.stats.KNNGraphValue;

import java.io.Closeable;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.HashMap;
import java.util.Map;

import static org.apache.lucene.codecs.CodecUtil.FOOTER_MAGIC;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.buildEngineFileName;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.calculateArraySize;
import static org.opensearch.knn.index.util.Faiss.FAISS_BINARY_INDEX_DESCRIPTION_PREFIX;

/**
 * This class writes the KNN docvalues to the segments
 */
@Log4j2
class KNN80DocValuesConsumer extends DocValuesConsumer implements Closeable {

    private final Logger logger = LogManager.getLogger(KNN80DocValuesConsumer.class);

    private final DocValuesConsumer delegatee;
    private final SegmentWriteState state;

    private static final Long CRC32_CHECKSUM_SANITY = 0xFFFFFFFF00000000L;

    KNN80DocValuesConsumer(DocValuesConsumer delegatee, SegmentWriteState state) {
        this.delegatee = delegatee;
        this.state = state;
    }

    @Override
    public void addBinaryField(FieldInfo field, DocValuesProducer valuesProducer) throws IOException {
        delegatee.addBinaryField(field, valuesProducer);
        if (isKNNBinaryFieldRequired(field)) {
            StopWatch stopWatch = new StopWatch();
            stopWatch.start();
            addKNNBinaryField(field, valuesProducer, false, true);
            stopWatch.stop();
            long time_in_millis = stopWatch.totalTime().millis();
            KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS.set(KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS.getValue() + time_in_millis);
            logger.warn("Refresh operation complete in " + time_in_millis + " ms");
        }
    }

    private boolean isKNNBinaryFieldRequired(FieldInfo field) {
        final KNNEngine knnEngine = getKNNEngine(field);
        log.debug(String.format("Read engine [%s] for field [%s]", knnEngine.getName(), field.getName()));
        return field.attributes().containsKey(KNNVectorFieldMapper.KNN_FIELD)
            && KNNEngine.getEnginesThatCreateCustomSegmentFiles().stream().anyMatch(engine -> engine == knnEngine);
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

    public void addKNNBinaryField(FieldInfo field, DocValuesProducer valuesProducer, boolean isMerge, boolean isRefresh)
        throws IOException {
        // Get values to be indexed
        BinaryDocValues values = valuesProducer.getBinary(field);
        if (KNNCodecUtil.getTotalLiveDocsCount(values) == 0) {
            return;
        }
        // Increment counter for number of graph index requests
        KNNCounter.GRAPH_INDEX_REQUESTS.increment();
        if (isMerge) {
            KNNGraphValue.MERGE_CURRENT_OPERATIONS.increment();
        }
        final KNNEngine knnEngine = getKNNEngine(field);
        final String engineFileName = buildEngineFileName(
            state.segmentInfo.name,
            knnEngine.getVersion(),
            field.name,
            knnEngine.getExtension()
        );
        final String indexPath = Paths.get(
            ((FSDirectory) (FilterDirectory.unwrap(state.directory))).getDirectory().toString(),
            engineFileName
        ).toString();

        state.directory.createOutput(engineFileName, state.context).close();
        boolean fromScratch = !field.attributes().containsKey(MODEL_ID);
        boolean iterative = fromScratch && KNNEngine.FAISS == knnEngine;
        createKNNIndex(field, values, knnEngine, indexPath, fromScratch, iterative, isMerge);

        if (isRefresh) {
            recordRefreshStats();
        }
        writeFooter(indexPath, engineFileName);
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

    private Map<String, Object> genParameters(boolean fromScratch, FieldInfo fieldInfo, KNNEngine knnEngine) throws IOException {
        Map<String, Object> parameters = new HashMap<>();
        ;
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

    private long initIndexFromScratch(long size, int dim, KNNEngine knnEngine, Map<String, Object> parameters) throws IOException {
        // Pass the path for the nms library to save the file
        return AccessController.doPrivileged((PrivilegedAction<Long>) () -> {
            return JNIService.initIndexFromScratch(size, dim, parameters, knnEngine);
        });
    }

    private void insertToIndex(KNNCodecUtil.VectorBatch pair, KNNEngine knnEngine, long indexAddress, Map<String, Object> parameters)
        throws IOException {
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

    private void createKNNIndexFromTemplate(
        FieldInfo field,
        BinaryDocValues values,
        KNNEngine knnEngine,
        String indexPath,
        Map<String, Object> parameters,
        boolean isMerge
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

        long arraySize = calculateArraySize(numDocs, batch.getDimension(), batch.serializationMode);

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
        Map<String, Object> parameters,
        boolean isMerge
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

        long arraySize = calculateArraySize(numDocs, batch.getDimension(), batch.serializationMode);

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
        Map<String, Object> parameters,
        boolean isMerge
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

        long arraySize = calculateArraySize(numDocs, batch.getDimension(), batch.serializationMode);

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

    private void createKNNIndex(
        FieldInfo fieldInfo,
        BinaryDocValues values,
        KNNEngine knnEngine,
        String indexPath,
        boolean fromScratch,
        boolean iterative,
        boolean isMerge
    ) throws IOException {
        Map<String, Object> parameters = genParameters(fromScratch, fieldInfo, knnEngine);
        if (fromScratch && iterative) {
            createKNNIndexFromScratchIteratively(fieldInfo, values, knnEngine, indexPath, parameters, isMerge);
        } else if (fromScratch) {
            createKNNIndexFromScratch(fieldInfo, values, knnEngine, indexPath, parameters, isMerge);
        } else {
            createKNNIndexFromTemplate(fieldInfo, values, knnEngine, indexPath, parameters, isMerge);
        }
        /*
        if(fromScratch) {
            createKNNIndexFromScratch(fieldInfo, values, knnEngine, indexPath, parameters, isMerge);
        } else {
            createKNNIndexFromTemplate(fieldInfo, values, knnEngine, indexPath, parameters, isMerge);
        }
        */
    }

    /**
     * Merges in the fields from the readers in mergeState
     *
     * @param mergeState Holds common state used during segment merging
     */
    @Override
    public void merge(MergeState mergeState) {
        try {
            delegatee.merge(mergeState);
            assert mergeState != null;
            assert mergeState.mergeFieldInfos != null;
            for (FieldInfo fieldInfo : mergeState.mergeFieldInfos) {
                DocValuesType type = fieldInfo.getDocValuesType();
                if (type == DocValuesType.BINARY && fieldInfo.attributes().containsKey(KNNVectorFieldMapper.KNN_FIELD)) {
                    StopWatch stopWatch = new StopWatch();
                    stopWatch.start();
                    addKNNBinaryField(fieldInfo, new KNN80DocValuesReader(mergeState), true, false);
                    stopWatch.stop();
                    long time_in_millis = stopWatch.totalTime().millis();
                    KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.set(KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.getValue() + time_in_millis);
                    logger.warn("Merge operation complete in " + time_in_millis + " ms");
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void addSortedSetField(FieldInfo field, DocValuesProducer valuesProducer) throws IOException {
        delegatee.addSortedSetField(field, valuesProducer);
    }

    @Override
    public void addSortedNumericField(FieldInfo field, DocValuesProducer valuesProducer) throws IOException {
        delegatee.addSortedNumericField(field, valuesProducer);
    }

    @Override
    public void addSortedField(FieldInfo field, DocValuesProducer valuesProducer) throws IOException {
        delegatee.addSortedField(field, valuesProducer);
    }

    @Override
    public void addNumericField(FieldInfo field, DocValuesProducer valuesProducer) throws IOException {
        delegatee.addNumericField(field, valuesProducer);
    }

    @Override
    public void close() throws IOException {
        delegatee.close();
    }

    @FunctionalInterface
    private interface NativeIndexCreator {
        void createIndex() throws IOException;
    }

    private void writeFooter(String indexPath, String engineFileName) throws IOException {
        // Opens the engine file that was created and appends a footer to it. The footer consists of
        // 1. A Footer magic number (int - 4 bytes)
        // 2. A checksum algorithm id (int - 4 bytes)
        // 3. A checksum (long - bytes)
        // The checksum is computed on all the bytes written to the file up to that point.
        // Logic where footer is written in Lucene can be found here:
        // https://github.com/apache/lucene/blob/branch_9_0/lucene/core/src/java/org/apache/lucene/codecs/CodecUtil.java#L390-L412
        OutputStream os = Files.newOutputStream(Paths.get(indexPath), StandardOpenOption.APPEND);
        ByteBuffer byteBuffer = ByteBuffer.allocate(8).order(ByteOrder.BIG_ENDIAN);
        byteBuffer.putInt(FOOTER_MAGIC);
        byteBuffer.putInt(0);
        os.write(byteBuffer.array());
        os.flush();

        ChecksumIndexInput checksumIndexInput = state.directory.openChecksumInput(engineFileName, state.context);
        checksumIndexInput.seek(checksumIndexInput.length());
        long value = checksumIndexInput.getChecksum();
        checksumIndexInput.close();

        if (isChecksumValid(value)) {
            throw new IllegalStateException("Illegal CRC-32 checksum: " + value + " (resource=" + os + ")");
        }

        // Write the CRC checksum to the end of the OutputStream and close the stream
        byteBuffer.putLong(0, value);
        os.write(byteBuffer.array());
        os.close();
    }

    private boolean isChecksumValid(long value) {
        // Check pulled from
        // https://github.com/apache/lucene/blob/branch_9_0/lucene/core/src/java/org/apache/lucene/codecs/CodecUtil.java#L644-L647
        return (value & CRC32_CHECKSUM_SANITY) != 0;
    }

    private VectorTransfer getVectorTransfer(VectorDataType vectorDataType) {
        if (VectorDataType.BINARY == vectorDataType) {
            return new VectorTransferByte(KNNSettings.getVectorStreamingMemoryLimit().getBytes());
        }
        return new VectorTransferFloat(KNNSettings.getVectorStreamingMemoryLimit().getBytes());
    }
}

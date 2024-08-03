/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Map;

import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.transfer.VectorTransfer;
import org.opensearch.knn.index.codec.transfer.VectorTransferByte;
import org.opensearch.knn.index.codec.transfer.VectorTransferFloat;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelCache;
import org.opensearch.knn.plugin.stats.KNNGraphValue;

import lombok.Builder;
import lombok.NonNull;
import lombok.Value;
import lombok.extern.log4j.Log4j2;

import static org.apache.lucene.codecs.CodecUtil.FOOTER_MAGIC;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.buildEngineFileName;

/**
 * Abstract class to build the KNN index and write it to disk
 */
@Log4j2
public abstract class NativeIndexWriter {
    private static final Long CRC32_CHECKSUM_SANITY = 0xFFFFFFFF00000000L;

    /**
     * Class that holds info about vectors
     */
    @Builder
    @Value
    protected static class NativeVectorInfo {
        private VectorDataType vectorDataType;
        private int dimension;
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

    /**
     * Gets the correct writer type from fieldInfo
     *
     * @param fieldInfo
     * @return correct NativeIndexWriter to make index specified in fieldInfo
     */
    public static NativeIndexWriter getWriter(FieldInfo fieldInfo) {
        final KNNEngine knnEngine = getKNNEngine(fieldInfo);
        boolean fromScratch = !fieldInfo.attributes().containsKey(MODEL_ID);
        boolean iterative = fromScratch && KNNEngine.FAISS == knnEngine;
        if (fromScratch && iterative) {
            return new NativeIndexWriterScratchIter();
        } else if (fromScratch) {
            return new NativeIndexWriterScratch();
        } else {
            return new NativeIndexWriterTemplate();
        }
    }

    /**
     * Method for creating a KNN index in the specified native library
     *
     * @param fieldInfo
     * @param valuesProducer
     * @param state
     * @param isMerge
     * @param isRefresh
     * @throws IOException
     */
    public void createKNNIndex(
        FieldInfo fieldInfo,
        DocValuesProducer valuesProducer,
        SegmentWriteState state,
        boolean isMerge,
        boolean isRefresh
    ) throws IOException {
        BinaryDocValues values = valuesProducer.getBinary(fieldInfo);
        if (KNNCodecUtil.getTotalLiveDocsCount(values) == 0) {
            log.debug("No live docs for field " + fieldInfo.name);
            return;
        }
        final KNNEngine knnEngine = getKNNEngine(fieldInfo);
        final String engineFileName = buildEngineFileName(
            state.segmentInfo.name,
            knnEngine.getVersion(),
            fieldInfo.name,
            knnEngine.getExtension()
        );
        final String indexPath = Paths.get(
            ((FSDirectory) (FilterDirectory.unwrap(state.directory))).getDirectory().toString(),
            engineFileName
        ).toString();

        state.directory.createOutput(engineFileName, state.context).close();
        NativeIndexInfo indexInfo = getIndexInfo(fieldInfo, valuesProducer, indexPath);
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
        writeFooter(indexPath, engineFileName, state);
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
     * @param valuesProducer
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

    private static KNNEngine getKNNEngine(@NonNull FieldInfo field) {
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

    private boolean isChecksumValid(long value) {
        // Check pulled from
        // https://github.com/apache/lucene/blob/branch_9_0/lucene/core/src/java/org/apache/lucene/codecs/CodecUtil.java#L644-L647
        return (value & CRC32_CHECKSUM_SANITY) != 0;
    }

    private void writeFooter(String indexPath, String engineFileName, SegmentWriteState state) throws IOException {
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
}

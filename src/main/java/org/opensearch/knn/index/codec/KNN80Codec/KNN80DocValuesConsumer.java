/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN80Codec;

import lombok.NonNull;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.ChecksumIndexInput;
import org.opensearch.common.StopWatch;
import org.opensearch.knn.index.codec.builder.KNNIndexBuilder;
import org.opensearch.knn.index.codec.builder.KNNIndexBuilderScratch;
import org.opensearch.knn.index.codec.builder.KNNIndexBuilderScratchIter;
import org.opensearch.knn.index.codec.builder.KNNIndexBuilderTemplate;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelCache;
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

import static org.apache.lucene.codecs.CodecUtil.FOOTER_MAGIC;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.buildEngineFileName;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.getTotalLiveDocsCount;

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
        if (getTotalLiveDocsCount(values) == 0) {
            logger.debug("No live docs for field " + field.name);
            return;
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

        KNNIndexBuilder indexBuilder = getBuilder(field, knnEngine);
        indexBuilder.setFieldInfo(field);
        indexBuilder.setIndexPath(indexPath);
        indexBuilder.setValuesProducer(valuesProducer);
        indexBuilder.setMerge(isMerge);
        indexBuilder.setRefresh(isRefresh);
        indexBuilder.createKNNIndex();

        writeFooter(indexPath, engineFileName);
    }

    public static KNNIndexBuilder getBuilder(FieldInfo fieldInfo, KNNEngine knnEngine) {
        boolean fromScratch = !fieldInfo.attributes().containsKey(MODEL_ID);
        boolean iterative = fromScratch && KNNEngine.FAISS == knnEngine;
        if (fromScratch && iterative) {
            return new KNNIndexBuilderScratchIter();
        } else if (fromScratch) {
            return new KNNIndexBuilderScratch();
        } else {
            return new KNNIndexBuilderTemplate();
        }
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
}

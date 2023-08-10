/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN80Codec;

import com.google.common.collect.ImmutableMap;
import lombok.NonNull;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.ChecksumIndexInput;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.knn.index.KNNSettings;
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
import static org.opensearch.knn.common.KNNConstants.FAISS_FLAT_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.FAISS_NSG_MIN_DOC_TO_FLAT;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_NSG;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.buildEngineFileName;

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
            addKNNBinaryField(field, valuesProducer);
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

    public void addKNNBinaryField(FieldInfo field, DocValuesProducer valuesProducer) throws IOException {
        // Get values to be indexed
        BinaryDocValues values = valuesProducer.getBinary(field);
        KNNCodecUtil.Pair pair = KNNCodecUtil.getFloats(values);
        if (pair.vectors.length == 0 || pair.docs.length == 0) {
            logger.info("Skipping engine index creation as there are no vectors or docs in the documents");
            return;
        }
        // Increment counter for number of graph index requests
        KNNCounter.GRAPH_INDEX_REQUESTS.increment();
        // Create library index either from model or from scratch
        String engineFileName;
        String indexPath;
        NativeIndexCreator indexCreator;
        final KNNEngine knnEngine = getKNNEngine(field);
        if (field.attributes().containsKey(MODEL_ID)) {

            String modelId = field.attributes().get(MODEL_ID);
            Model model = ModelCache.getInstance().get(modelId);

            engineFileName = buildEngineFileName(state.segmentInfo.name, knnEngine.getVersion(), field.name, knnEngine.getExtension());
            indexPath = Paths.get(((FSDirectory) (FilterDirectory.unwrap(state.directory))).getDirectory().toString(), engineFileName)
                .toString();

            if (model.getModelBlob() == null) {
                throw new RuntimeException("There is no trained model with id \"" + modelId + "\"");
            }

            indexCreator = () -> createKNNIndexFromTemplate(model.getModelBlob(), pair, knnEngine, indexPath);
        } else {

            engineFileName = buildEngineFileName(state.segmentInfo.name, knnEngine.getVersion(), field.name, knnEngine.getExtension());
            indexPath = Paths.get(((FSDirectory) (FilterDirectory.unwrap(state.directory))).getDirectory().toString(), engineFileName)
                .toString();

            indexCreator = () -> createKNNIndexFromScratch(field, pair, knnEngine, indexPath);
        }

        // This is a bit of a hack. We have to create an output here and then immediately close it to ensure that
        // engineFileName is added to the tracked files by Lucene's TrackingDirectoryWrapper. Otherwise, the file will
        // not be marked as added to the directory.
        state.directory.createOutput(engineFileName, state.context).close();
        indexCreator.createIndex();
        writeFooter(indexPath, engineFileName);
    }

    private void createKNNIndexFromTemplate(byte[] model, KNNCodecUtil.Pair pair, KNNEngine knnEngine, String indexPath) {
        Map<String, Object> parameters = ImmutableMap.of(
            KNNConstants.INDEX_THREAD_QTY,
            KNNSettings.state().getSettingValue(KNNSettings.KNN_ALGO_PARAM_INDEX_THREAD_QTY)
        );
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            JNIService.createIndexFromTemplate(pair.docs, pair.vectors, indexPath, model, parameters, knnEngine.getName());
            return null;
        });
    }

    private void createKNNIndexFromScratch(FieldInfo fieldInfo, KNNCodecUtil.Pair pair, KNNEngine knnEngine, String indexPath)
        throws IOException {
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

        // Used to determine how many threads to use when indexing
        parameters.put(KNNConstants.INDEX_THREAD_QTY, KNNSettings.state().getSettingValue(KNNSettings.KNN_ALGO_PARAM_INDEX_THREAD_QTY));
        Object name = parameters.get(NAME);
        if (name != null && name.equals(METHOD_NSG)) {
            /**
             * TODO:
             * Search params not supported for the NSG index
             */
            if (parameters.containsKey(PARAMETERS)) {
                parameters.remove(PARAMETERS);
            }
            /** TODO:
             * when numIds is too small, NSG graph would core/throw exception
             * because There are too much invalid entries in the knn graph.
             */
            if (pair.docs.length < FAISS_NSG_MIN_DOC_TO_FLAT) {
                parameters.put(INDEX_DESCRIPTION_PARAMETER, FAISS_FLAT_DESCRIPTION);
            }
        }
        log.debug(String.format("docSize:[%d], parameters:[%s]", pair.docs.length, parameters.toString()));

        // Pass the path for the nms library to save the file
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            JNIService.createIndex(pair.docs, pair.vectors, indexPath, parameters, knnEngine.getName());
            return null;
        });
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
                    addKNNBinaryField(fieldInfo, new KNN80DocValuesReader(mergeState));
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
}

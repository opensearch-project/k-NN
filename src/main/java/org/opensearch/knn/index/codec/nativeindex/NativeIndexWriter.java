/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.AllArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.opensearch.common.Nullable;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.index.util.IndexUtil;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelCache;
import org.opensearch.knn.plugin.stats.KNNGraphValue;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.Map;

import static org.apache.lucene.codecs.CodecUtil.FOOTER_MAGIC;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.opensearch.knn.common.FieldInfoExtractor.extractKNNEngine;
import static org.opensearch.knn.common.FieldInfoExtractor.extractVectorDataType;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNVectorUtil.iterateVectorValuesOnce;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.buildEngineFileName;
import static org.opensearch.knn.index.engine.faiss.Faiss.FAISS_BINARY_INDEX_DESCRIPTION_PREFIX;

/**
 * Writes KNN Index for a field in a segment. This is intended to be used for native engines
 */
@AllArgsConstructor
@Log4j2
public class NativeIndexWriter {
    private static final Long CRC32_CHECKSUM_SANITY = 0xFFFFFFFF00000000L;

    private final SegmentWriteState state;
    private final FieldInfo fieldInfo;
    private final NativeIndexBuildStrategy indexBuilder;
    @Nullable
    private final QuantizationState quantizationState;

    /**
     * Gets the correct writer type from fieldInfo
     *
     * @param fieldInfo
     * @return correct NativeIndexWriter to make index specified in fieldInfo
     */
    public static NativeIndexWriter getWriter(final FieldInfo fieldInfo, SegmentWriteState state) {
        return createWriter(fieldInfo, state, null);
    }

    /**
     * Gets the correct writer type for the specified field, using a given QuantizationModel.
     *
     * This method returns a NativeIndexWriter instance that is tailored to the specific characteristics
     * of the field described by the provided FieldInfo. It determines whether to use a template-based
     * writer or an iterative approach based on the engine type and whether the field is associated with a template.
     *
     * If quantization is required, the QuantizationModel is passed to the writer to facilitate the quantization process.
     *
     * @param fieldInfo          The FieldInfo object containing metadata about the field for which the writer is needed.
     * @param state              The SegmentWriteState representing the current segment's writing context.
     * @param quantizationState  The QuantizationState that contains  quantization state required for quantization
     * @return                   A NativeIndexWriter instance appropriate for the specified field, configured with or without quantization.
     */
    public static NativeIndexWriter getWriter(
        final FieldInfo fieldInfo,
        final SegmentWriteState state,
        final QuantizationState quantizationState
    ) {
        return createWriter(fieldInfo, state, quantizationState);
    }

    /**
     * flushes the index
     *
     * @param knnVectorValues
     * @throws IOException
     */
    public void flushIndex(final KNNVectorValues<?> knnVectorValues) throws IOException {
        iterateVectorValuesOnce(knnVectorValues);
        buildAndWriteIndex(knnVectorValues);
        recordRefreshStats();
    }

    /**
     * Merges kNN index
     * @param knnVectorValues
     * @throws IOException
     */
    public void mergeIndex(final KNNVectorValues<?> knnVectorValues) throws IOException {
        iterateVectorValuesOnce(knnVectorValues);
        if (knnVectorValues.docId() == NO_MORE_DOCS) {
            // This is in place so we do not add metrics
            log.debug("Skipping mergeIndex, vector values are already iterated for {}", fieldInfo.name);
            return;
        }

        long bytesPerVector = knnVectorValues.bytesPerVector();
        startMergeStats((int) knnVectorValues.totalLiveDocs(), bytesPerVector);
        buildAndWriteIndex(knnVectorValues);
        endMergeStats((int) knnVectorValues.totalLiveDocs(), bytesPerVector);
    }

    private void buildAndWriteIndex(final KNNVectorValues<?> knnVectorValues) throws IOException {
        if (knnVectorValues.totalLiveDocs() == 0) {
            log.debug("No live docs for field " + fieldInfo.name);
            return;
        }

        final KNNEngine knnEngine = extractKNNEngine(fieldInfo);
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

        final BuildIndexParams nativeIndexParams = indexParams(fieldInfo, indexPath, knnEngine);
        indexBuilder.buildAndWriteIndex(nativeIndexParams, knnVectorValues);
        writeFooter(indexPath, engineFileName, state);
    }

    // The logic for building parameters need to be cleaned up. There are various cases handled here
    // Currently it falls under two categories - with model and without model. Without model is further divided based on vector data type
    // TODO: Refactor this so its scalable. Possibly move it out of this class
    private BuildIndexParams indexParams(FieldInfo fieldInfo, String indexPath, KNNEngine knnEngine) throws IOException {
        final Map<String, Object> parameters;
        VectorDataType vectorDataType;
        if (quantizationState != null) {
            vectorDataType = QuantizationService.getInstance().getVectorDataTypeForTransfer(fieldInfo);
        } else {
            vectorDataType = extractVectorDataType(fieldInfo);
        }
        if (fieldInfo.attributes().containsKey(MODEL_ID)) {
            Model model = getModel(fieldInfo);
            parameters = getTemplateParameters(fieldInfo, model);
        } else {
            parameters = getParameters(fieldInfo, vectorDataType, knnEngine);
        }

        return BuildIndexParams.builder()
            .fieldName(fieldInfo.name)
            .parameters(parameters)
            .vectorDataType(vectorDataType)
            .knnEngine(knnEngine)
            .indexPath(indexPath)
            .quantizationState(quantizationState)
            .build();
    }

    private Map<String, Object> getParameters(FieldInfo fieldInfo, VectorDataType vectorDataType, KNNEngine knnEngine) throws IOException {
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

        parameters.put(KNNConstants.VECTOR_DATA_TYPE_FIELD, vectorDataType.getValue());
        // In OpenSearch 2.16, we added the prefix for binary indices in the index description in the codec logic.
        // After 2.16, we added the binary prefix in the faiss library code. However, to ensure backwards compatibility,
        // we need to ensure that if the description does not contain the prefix but the type is binary, we add the
        // description.
        maybeAddBinaryPrefixForFaissBWC(knnEngine, parameters, fieldAttributes);

        // Used to determine how many threads to use when indexing
        parameters.put(KNNConstants.INDEX_THREAD_QTY, KNNSettings.state().getSettingValue(KNNSettings.KNN_ALGO_PARAM_INDEX_THREAD_QTY));

        return parameters;
    }

    private void maybeAddBinaryPrefixForFaissBWC(KNNEngine knnEngine, Map<String, Object> parameters, Map<String, String> fieldAttributes) {
        if (KNNEngine.FAISS != knnEngine) {
            return;
        }

        if (!VectorDataType.BINARY.getValue()
            .equals(fieldAttributes.getOrDefault(KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.DEFAULT.getValue()))) {
            return;
        }

        if (parameters.get(KNNConstants.INDEX_DESCRIPTION_PARAMETER) == null) {
            return;
        }

        if (parameters.get(KNNConstants.INDEX_DESCRIPTION_PARAMETER).toString().startsWith(FAISS_BINARY_INDEX_DESCRIPTION_PREFIX)) {
            return;
        }

        parameters.put(
            KNNConstants.INDEX_DESCRIPTION_PARAMETER,
            FAISS_BINARY_INDEX_DESCRIPTION_PREFIX + parameters.get(KNNConstants.INDEX_DESCRIPTION_PARAMETER).toString()
        );
        IndexUtil.updateVectorDataTypeToParameters(parameters, VectorDataType.BINARY);
    }

    private Map<String, Object> getTemplateParameters(FieldInfo fieldInfo, Model model) throws IOException {
        Map<String, Object> parameters = new HashMap<>();
        parameters.put(KNNConstants.INDEX_THREAD_QTY, KNNSettings.state().getSettingValue(KNNSettings.KNN_ALGO_PARAM_INDEX_THREAD_QTY));
        parameters.put(KNNConstants.MODEL_ID, fieldInfo.attributes().get(MODEL_ID));
        parameters.put(KNNConstants.MODEL_BLOB_PARAMETER, model.getModelBlob());
        if (FieldInfoExtractor.extractQuantizationConfig(fieldInfo) != QuantizationConfig.EMPTY) {
            IndexUtil.updateVectorDataTypeToParameters(parameters, VectorDataType.BINARY);
        } else {
            IndexUtil.updateVectorDataTypeToParameters(parameters, model.getModelMetadata().getVectorDataType());
        }

        return parameters;
    }

    private Model getModel(FieldInfo fieldInfo) {
        String modelId = fieldInfo.attributes().get(MODEL_ID);
        Model model = ModelCache.getInstance().get(modelId);
        if (model.getModelBlob() == null) {
            throw new RuntimeException(String.format("There is no trained model with id \"%s\"", modelId));
        }
        return model;
    }

    private void startMergeStats(int numDocs, long bytesPerVector) {
        KNNGraphValue.MERGE_CURRENT_OPERATIONS.increment();
        KNNGraphValue.MERGE_CURRENT_DOCS.incrementBy(numDocs);
        KNNGraphValue.MERGE_CURRENT_SIZE_IN_BYTES.incrementBy(bytesPerVector);
        KNNGraphValue.MERGE_TOTAL_OPERATIONS.increment();
        KNNGraphValue.MERGE_TOTAL_DOCS.incrementBy(numDocs);
        KNNGraphValue.MERGE_TOTAL_SIZE_IN_BYTES.incrementBy(bytesPerVector);
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

    /**
     * Helper method to create the appropriate NativeIndexWriter based on the field info and quantization state.
     *
     * @param fieldInfo          The FieldInfo object containing metadata about the field for which the writer is needed.
     * @param state              The SegmentWriteState representing the current segment's writing context.
     * @param quantizationState  The QuantizationState that contains quantization state required for quantization, can be null.
     * @return                   A NativeIndexWriter instance appropriate for the specified field, configured with or without quantization.
     */
    private static NativeIndexWriter createWriter(
        final FieldInfo fieldInfo,
        final SegmentWriteState state,
        @Nullable final QuantizationState quantizationState
    ) {
        final KNNEngine knnEngine = extractKNNEngine(fieldInfo);
        boolean isTemplate = fieldInfo.attributes().containsKey(MODEL_ID);
        boolean iterative = !isTemplate && KNNEngine.FAISS == knnEngine;
        NativeIndexBuildStrategy strategy = iterative
            ? MemOptimizedNativeIndexBuildStrategy.getInstance()
            : DefaultIndexBuildStrategy.getInstance();
        return new NativeIndexWriter(state, fieldInfo, strategy, quantizationState);
    }
}

/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.common.StopWatch;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexWriter;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.plugin.stats.KNNGraphValue;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.opensearch.knn.common.FieldInfoExtractor.extractVectorDataType;

/**
 * A KNNVectorsWriter class for writing the vector data strcutures and flat vectors for Native Engines.
 */
@Log4j2
public class NativeEngines990KnnVectorsWriter extends KnnVectorsWriter {
    private static final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(NativeEngines990KnnVectorsWriter.class);

    private static final String FLUSH_OPERATION = "flush";
    private static final String MERGE_OPERATION = "merge";

    private final SegmentWriteState segmentWriteState;
    private final FlatVectorsWriter flatVectorsWriter;
    private KNN990QuantizationStateWriter quantizationStateWriter;
    private final List<NativeEngineFieldVectorsWriter<?>> fields = new ArrayList<>();
    private boolean finished;
    private final QuantizationService quantizationService = QuantizationService.getInstance();

    public NativeEngines990KnnVectorsWriter(SegmentWriteState segmentWriteState, FlatVectorsWriter flatVectorsWriter) {
        this.segmentWriteState = segmentWriteState;
        this.flatVectorsWriter = flatVectorsWriter;
    }

    /**
     * Add new field for indexing.
     * @param fieldInfo {@link FieldInfo}
     */
    @Override
    public KnnFieldVectorsWriter<?> addField(final FieldInfo fieldInfo) throws IOException {
        final NativeEngineFieldVectorsWriter<?> newField = NativeEngineFieldVectorsWriter.create(fieldInfo, segmentWriteState.infoStream);
        fields.add(newField);
        return flatVectorsWriter.addField(fieldInfo, newField);
    }

    /**
     * Flush all buffered data on disk. This is not fsync. This is lucene flush.
     *
     * @param maxDoc int
     * @param sortMap {@link Sorter.DocMap}
     */
    @Override
    public void flush(int maxDoc, final Sorter.DocMap sortMap) throws IOException {
        flatVectorsWriter.flush(maxDoc, sortMap);

        for (final NativeEngineFieldVectorsWriter<?> field : fields) {
            trainAndIndex(
                field.getFieldInfo(),
                (vectorDataType, fieldInfo, fieldVectorsWriter) -> getKNNVectorValues(vectorDataType, fieldVectorsWriter),
                NativeIndexWriter::flushIndex,
                field,
                KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS,
                FLUSH_OPERATION
            );
        }
    }

    @Override
    public void mergeOneField(final FieldInfo fieldInfo, final MergeState mergeState) throws IOException {
        // This will ensure that we are merging the FlatIndex during force merge.
        flatVectorsWriter.mergeOneField(fieldInfo, mergeState);

        // For merge, pick values from flat vector and reindex again. This will use the flush operation to create graphs
        trainAndIndex(
            fieldInfo,
            this::getKNNVectorValuesForMerge,
            NativeIndexWriter::mergeIndex,
            mergeState,
            KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS,
            MERGE_OPERATION
        );
    }

    /**
     * Called once at the end before close
     */
    @Override
    public void finish() throws IOException {
        if (finished) {
            throw new IllegalStateException("NativeEnginesKNNVectorsWriter is already finished");
        }
        finished = true;
        if (quantizationStateWriter != null) {
            quantizationStateWriter.writeFooter();
        }
        flatVectorsWriter.finish();
    }

    /**
     * Closes this stream and releases any system resources associated
     * with it. If the stream is already closed then invoking this
     * method has no effect.
     *
     * <p> As noted in {@link AutoCloseable#close()}, cases where the
     * close may fail require careful attention. It is strongly advised
     * to relinquish the underlying resources and to internally
     * <em>mark</em> the {@code Closeable} as closed, prior to throwing
     * the {@code IOException}.
     *
     * @throws IOException if an I/O error occurs
     */
    @Override
    public void close() throws IOException {
        if (quantizationStateWriter != null) {
            quantizationStateWriter.closeOutput();
        }
        IOUtils.close(flatVectorsWriter);
    }

    /**
     * Return the memory usage of this object in bytes. Negative values are illegal.
     */
    @Override
    public long ramBytesUsed() {
        return SHALLOW_SIZE + flatVectorsWriter.ramBytesUsed() + fields.stream()
            .mapToLong(NativeEngineFieldVectorsWriter::ramBytesUsed)
            .sum();
    }

    /**
     * Retrieves the {@link KNNVectorValues} for a specific field based on the vector data type and field writer.
     *
     * @param vectorDataType The {@link VectorDataType} representing the type of vectors stored.
     * @param field          The {@link NativeEngineFieldVectorsWriter} representing the field from which to retrieve vectors.
     * @param <T>            The type of vectors being processed.
     * @return The {@link KNNVectorValues} associated with the field.
     */
    private <T> KNNVectorValues<T> getKNNVectorValues(final VectorDataType vectorDataType, final NativeEngineFieldVectorsWriter<?> field) {
        return (KNNVectorValues<T>) KNNVectorValuesFactory.getVectorValues(vectorDataType, field.getDocsWithField(), field.getVectors());
    }

    /**
     * Retrieves the {@link KNNVectorValues} for a specific field during a merge operation, based on the vector data type.
     *
     * @param vectorDataType The {@link VectorDataType} representing the type of vectors stored.
     * @param fieldInfo      The {@link FieldInfo} object containing metadata about the field.
     * @param mergeState     The {@link MergeState} representing the state of the merge operation.
     * @param <T>            The type of vectors being processed.
     * @return The {@link KNNVectorValues} associated with the field during the merge.
     * @throws IOException If an I/O error occurs during the retrieval.
     */
    private <T> KNNVectorValues<T> getKNNVectorValuesForMerge(
        final VectorDataType vectorDataType,
        final FieldInfo fieldInfo,
        final MergeState mergeState
    ) throws IOException {
        switch (fieldInfo.getVectorEncoding()) {
            case FLOAT32:
                FloatVectorValues mergedFloats = MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
                return (KNNVectorValues<T>) KNNVectorValuesFactory.getVectorValues(vectorDataType, mergedFloats);
            case BYTE:
                ByteVectorValues mergedBytes = MergedVectorValues.mergeByteVectorValues(fieldInfo, mergeState);
                return (KNNVectorValues<T>) KNNVectorValuesFactory.getVectorValues(vectorDataType, mergedBytes);
            default:
                throw new IllegalStateException("Unsupported vector encoding [" + fieldInfo.getVectorEncoding() + "]");
        }
    }

    /**
     * Functional interface representing an operation that indexes the provided {@link KNNVectorValues}.
     *
     * @param <T> The type of vectors being processed.
     */
    @FunctionalInterface
    private interface IndexOperation<T> {
        void buildAndWrite(NativeIndexWriter writer, KNNVectorValues<T> knnVectorValues, int totalLiveDocs) throws IOException;
    }

    /**
     * Functional interface representing a method that retrieves {@link KNNVectorValues} based on
     * the vector data type, field information, and the merge state.
     *
     * @param <DataType>   The type of the data representing the vector (e.g., {@link VectorDataType}).
     * @param <FieldInfo>  The metadata about the field.
     * @param <MergeState> The state of the merge operation.
     * @param <Result>     The result of the retrieval, typically {@link KNNVectorValues}.
     */
    @FunctionalInterface
    private interface VectorValuesRetriever<DataType, FieldInfo, MergeState, Result> {
        Result apply(DataType vectorDataType, FieldInfo fieldInfo, MergeState mergeState) throws IOException;
    }

    /**
     * Unified method for processing a field during either the indexing or merge operation. This method retrieves vector values
     * based on the provided vector data type and applies the specified index operation, potentially including quantization if needed.
     *
     * @param fieldInfo              The {@link FieldInfo} object containing metadata about the field.
     * @param vectorValuesRetriever  A functional interface that retrieves {@link KNNVectorValues} based on the vector data type,
     *                                field information, and additional context (e.g., merge state or field writer).
     * @param indexOperation         A functional interface that performs the indexing operation using the retrieved
     *                                {@link KNNVectorValues}.
     * @param VectorProcessingContext                The additional context required for retrieving the vector values (e.g., {@link MergeState} or {@link NativeEngineFieldVectorsWriter}).
     *                                               From Flush we need NativeFieldWriter which contains total number of vectors while from Merge we need merge state which contains vector information
     * @param <T>                    The type of vectors being processed.
     * @param <C>                    The type of the context needed for retrieving the vector values.
     * @throws IOException If an I/O error occurs during the processing.
     */
    private <T, C> void trainAndIndex(
        final FieldInfo fieldInfo,
        final VectorValuesRetriever<VectorDataType, FieldInfo, C, KNNVectorValues<T>> vectorValuesRetriever,
        final IndexOperation<T> indexOperation,
        final C VectorProcessingContext,
        final KNNGraphValue graphBuildTime,
        final String operationName
    ) throws IOException {
        final VectorDataType vectorDataType = extractVectorDataType(fieldInfo);
        KNNVectorValues<T> knnVectorValues = vectorValuesRetriever.apply(vectorDataType, fieldInfo, VectorProcessingContext);
        QuantizationParams quantizationParams = quantizationService.getQuantizationParams(fieldInfo);
        QuantizationState quantizationState = null;
        // Count the docIds
        int totalLiveDocs = getLiveDocs(vectorValuesRetriever.apply(vectorDataType, fieldInfo, VectorProcessingContext));
        if (quantizationParams != null && totalLiveDocs > 0) {
            initQuantizationStateWriterIfNecessary();
            quantizationState = quantizationService.train(quantizationParams, knnVectorValues, totalLiveDocs);
            quantizationStateWriter.writeState(fieldInfo.getFieldNumber(), quantizationState);
        }
        NativeIndexWriter writer = (quantizationParams != null)
            ? NativeIndexWriter.getWriter(fieldInfo, segmentWriteState, quantizationState)
            : NativeIndexWriter.getWriter(fieldInfo, segmentWriteState);

        knnVectorValues = vectorValuesRetriever.apply(vectorDataType, fieldInfo, VectorProcessingContext);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        indexOperation.buildAndWrite(writer, knnVectorValues, totalLiveDocs);
        long time_in_millis = stopWatch.totalTime().millis();
        graphBuildTime.incrementBy(time_in_millis);
        log.warn("Graph build took " + time_in_millis + " ms for " + operationName);
    }

    /**
     * The {@link KNNVectorValues} will be exhausted after this function run. So make sure that you are not sending the
     * vectorsValues object which you plan to use later
     */
    private int getLiveDocs(KNNVectorValues<?> vectorValues) throws IOException {
        // Count all the live docs as there vectorValues.totalLiveDocs() just gives the cost for the FloatVectorValues,
        // and doesn't tell the correct number of docs, if there are deleted docs in the segment. So we are counting
        // the total live docs here.
        int liveDocs = 0;
        while (vectorValues.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
            liveDocs++;
        }
        return liveDocs;
    }

    private void initQuantizationStateWriterIfNecessary() throws IOException {
        if (quantizationStateWriter == null) {
            quantizationStateWriter = new KNN990QuantizationStateWriter(segmentWriteState);
            quantizationStateWriter.writeHeader(segmentWriteState);
        }
    }
}

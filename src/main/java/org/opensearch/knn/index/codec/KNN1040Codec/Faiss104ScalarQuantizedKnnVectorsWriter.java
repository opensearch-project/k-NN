/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.common.StopWatch;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngineFieldVectorsWriter;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsWriter;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexWriter;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.plugin.stats.KNNGraphValue;

import java.io.IOException;
import java.util.function.Supplier;

import static org.opensearch.knn.common.FieldInfoExtractor.extractVectorDataType;
import static org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory.getKNNVectorValuesSupplierForMerge;
import static org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory.getVectorValuesSupplier;

/**
 * Writer for Faiss BBQ vector fields. Unlike {@link org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsWriter}
 * which handles multiple fields, this writer handles exactly one field per format instance
 * (since each BBQ field gets its own dedicated format via per-field routing).
 *
 * <p>Write path:
 * <ol>
 *   <li>Flat vectors are written by Lucene's BBQ flat writer (.vec + .veq/.vemq files)</li>
 *   <li>HNSW graph is built by native Faiss via {@link NativeIndexWriter} (.faiss file)</li>
 * </ol>
 *
 * <p>No quantization training is needed — Lucene's flat format handles quantization internally.
 */
@Log4j2
class Faiss104ScalarQuantizedKnnVectorsWriter extends KnnVectorsWriter {
    private static final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(Faiss104ScalarQuantizedKnnVectorsWriter.class);

    private final SegmentWriteState segmentWriteState;
    private final FlatVectorsWriter flatVectorsWriter;
    private final int approximateThreshold;
    // Single field — BBQ gets a dedicated format per field via BasePerFieldKnnVectorsFormat
    private NativeEngineFieldVectorsWriter<?> field;
    private boolean finished;

    Faiss104ScalarQuantizedKnnVectorsWriter(
        SegmentWriteState segmentWriteState,
        FlatVectorsWriter flatVectorsWriter,
        int approximateThreshold
    ) {
        this.segmentWriteState = segmentWriteState;
        this.flatVectorsWriter = flatVectorsWriter;
        this.approximateThreshold = approximateThreshold;
    }

    /**
     * Only one field is expected per format instance. Throws if called more than once.
     */
    @Override
    public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
        if (this.field != null) {
            throw new IllegalStateException(
                Faiss104ScalarQuantizedKnnVectorsWriter.class.getSimpleName()
                    + " supports only a single field, but addField was called for ["
                    + fieldInfo.name
                    + "] after ["
                    + this.field.getFieldInfo().name
                    + "]"
            );
        }
        this.field = NativeEngineFieldVectorsWriter.create(fieldInfo, flatVectorsWriter.addField(fieldInfo), segmentWriteState.infoStream);
        return field;
    }

    /**
     * Flushes flat vectors first (Lucene BBQ format), then builds the native HNSW graph
     * if the doc count meets the approximate threshold.
     */
    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        // Always flush flat vectors — these are the quantized + raw vectors managed by Lucene
        flatVectorsWriter.flush(maxDoc, sortMap);

        if (field == null) {
            return;
        }

        final FieldInfo fieldInfo = field.getFieldInfo();
        int totalLiveDocs = field.getVectors().size();
        if (totalLiveDocs == 0) {
            log.debug("[Flush] No live docs for field {}", fieldInfo.getName());
            return;
        }

        // Skip HNSW graph building when below threshold; flat vectors are still available for exact search
        if (NativeEngines990KnnVectorsWriter.shouldSkipBuildingVectorDataStructure(totalLiveDocs, approximateThreshold)) {
            log.debug(
                "Skip building vector data structure for field: {}, as liveDoc: {} is less than the threshold {} during flush",
                fieldInfo.name,
                totalLiveDocs,
                approximateThreshold
            );
            return;
        }

        final VectorDataType vectorDataType = extractVectorDataType(fieldInfo);
        final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier = getVectorValuesSupplier(
            vectorDataType,
            field.getFlatFieldVectorsWriter().getDocsWithFieldSet(),
            field.getVectors()
        );

        final NativeIndexWriter writer = NativeIndexWriter.getWriter(fieldInfo, segmentWriteState);

        StopWatch stopWatch = new StopWatch().start();
        writer.flushIndex(knnVectorValuesSupplier, totalLiveDocs);
        long timeInMillis = stopWatch.stop().totalTime().millis();
        KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS.incrementBy(timeInMillis);
        log.debug("Flush took {} ms for vector field [{}]", timeInMillis, fieldInfo.getName());
    }

    /**
     * Merges flat vectors first, then builds the native HNSW graph for the merged segment.
     */
    @Override
    public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        // Merge flat index first — required even if we skip HNSW graph building
        flatVectorsWriter.mergeOneField(fieldInfo, mergeState);

        final VectorDataType vectorDataType = extractVectorDataType(fieldInfo);
        final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier = getKNNVectorValuesSupplierForMerge(
            vectorDataType,
            fieldInfo,
            mergeState
        );
        int totalLiveDocs = NativeEngines990KnnVectorsWriter.getLiveDocs(knnVectorValuesSupplier.get());
        if (totalLiveDocs == 0) {
            log.debug("[Merge] No live docs for field {}", fieldInfo.getName());
            return;
        }

        if (NativeEngines990KnnVectorsWriter.shouldSkipBuildingVectorDataStructure(totalLiveDocs, approximateThreshold)) {
            log.debug(
                "Skip building vector data structure for field: {}, as liveDoc: {} is less than the threshold {} during merge",
                fieldInfo.name,
                totalLiveDocs,
                approximateThreshold
            );
            return;
        }

        final NativeIndexWriter writer = NativeIndexWriter.getWriter(fieldInfo, segmentWriteState);

        StopWatch stopWatch = new StopWatch().start();
        writer.mergeIndex(knnVectorValuesSupplier, totalLiveDocs);
        long timeInMillis = stopWatch.stop().totalTime().millis();
        KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.incrementBy(timeInMillis);
        log.debug("Merge took {} ms for vector field [{}]", timeInMillis, fieldInfo.getName());
    }

    @Override
    public void finish() throws IOException {
        if (finished) {
            throw new IllegalStateException(Faiss104ScalarQuantizedKnnVectorsWriter.class.getSimpleName() + " is already finished");
        }
        finished = true;
        // No quantizationStateWriter to finalize — BBQ doesn't use the k-NN quantization framework
        flatVectorsWriter.finish();
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(flatVectorsWriter);
    }

    @Override
    public long ramBytesUsed() {
        return SHALLOW_SIZE + flatVectorsWriter.ramBytesUsed() + (field != null ? field.ramBytesUsed() : 0);
    }
}

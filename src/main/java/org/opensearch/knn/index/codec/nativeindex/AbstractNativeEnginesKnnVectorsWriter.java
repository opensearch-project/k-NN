/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.lucene104.QuantizedByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.common.Nullable;
import org.opensearch.common.StopWatch;
import org.opensearch.common.TriFunction;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.plugin.stats.KNNGraphValue;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static org.opensearch.knn.common.FieldInfoExtractor.extractVectorDataType;
import static org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory.getKNNVectorValuesSupplierForMerge;
import static org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory.getVectorValuesSupplier;

@Log4j2
public abstract class AbstractNativeEnginesKnnVectorsWriter extends KnnVectorsWriter {
    protected <T> void doFlush(
        final FieldInfo fieldInfo,
        final FlatFieldVectorsWriter<?> fieldWriter,
        final T vectors,
        @Nullable final TriFunction<FieldInfo, Supplier<KNNVectorValues<?>>, Integer, QuantizationState> quantizationStateSupplier,
        final Integer approximateThreshold,
        final SegmentWriteState segmentWriteState,
        final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory,
        @Nullable final QuantizedByteVectorValues quantizedByteVectorValues
    ) throws IOException {
        // Check total live docs first to avoid unnecessary supplier creation for empty fields
        final int totalLiveDocs;
        // NativeEngines990KnnVectorsWriter keeps vector as Map<DocId, Vector> as BQ sampler needs random access.
        // while Faiss1040ScalarQuantizedKnnVectorsWriter keeps vector as List<Vector> from FlatFieldVectorsWriter.
        if (vectors instanceof Map) {
            totalLiveDocs = ((Map) vectors).size();
        } else {
            totalLiveDocs = ((List) vectors).size();
        }

        if (totalLiveDocs == 0) {
            log.debug("[Flush] No live docs for field {}", fieldInfo.getName());
            return;
        }

        // Get vector values supplier
        final VectorDataType vectorDataType = extractVectorDataType(fieldInfo);
        final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier;
        if (vectors instanceof Map) {
            knnVectorValuesSupplier = getVectorValuesSupplier(vectorDataType, fieldWriter.getDocsWithFieldSet(), (Map) vectors);
        } else {
            knnVectorValuesSupplier = getVectorValuesSupplier(vectorDataType, fieldWriter.getDocsWithFieldSet(), (List) vectors);
        }

        QuantizationState quantizationState = null;
        if (quantizationStateSupplier != null) {
            // should skip graph building only for non quantization use case and if threshold is met
            quantizationState = quantizationStateSupplier.apply(fieldInfo, knnVectorValuesSupplier, totalLiveDocs);
            if (quantizationState == null && shouldSkipBuildingVectorDataStructure(totalLiveDocs, approximateThreshold)) {
                log.debug(
                    "Skip building vector data structure for field: {}, as liveDoc: {} is less than the threshold {} during flush",
                    fieldInfo.name,
                    totalLiveDocs,
                    approximateThreshold
                );
                return;
            }
        }

        final NativeIndexWriter writer = NativeIndexWriter.getWriter(
            fieldInfo,
            segmentWriteState,
            quantizationState,
            nativeIndexBuildStrategyFactory,
            quantizedByteVectorValues
        );

        final StopWatch stopWatch = new StopWatch().start();
        writer.flushIndex(knnVectorValuesSupplier, totalLiveDocs);
        final long time_in_millis = stopWatch.stop().totalTime().millis();
        KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS.incrementBy(time_in_millis);
        log.debug("Flush took {} ms for vector field [{}]", time_in_millis, fieldInfo.getName());
    }

    protected void doMergeOneField(
        final FieldInfo fieldInfo,
        final MergeState mergeState,
        @Nullable final TriFunction<FieldInfo, Supplier<KNNVectorValues<?>>, Integer, QuantizationState> quantizationStateSupplier,
        final Integer approximateThreshold,
        final SegmentWriteState segmentWriteState,
        final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory,
        @Nullable final QuantizedByteVectorValues quantizedByteVectorValues
    ) throws IOException {
        final VectorDataType vectorDataType = extractVectorDataType(fieldInfo);
        final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier = getKNNVectorValuesSupplierForMerge(
            vectorDataType,
            fieldInfo,
            mergeState
        );
        final int totalLiveDocs = getLiveDocs(knnVectorValuesSupplier.get());
        if (totalLiveDocs == 0) {
            log.debug("[Merge] No live docs for field {}", fieldInfo.getName());
            return;
        }

        QuantizationState quantizationState = null;
        if (quantizationStateSupplier != null) {
            quantizationState = quantizationStateSupplier.apply(fieldInfo, knnVectorValuesSupplier, totalLiveDocs);
            // should skip graph building only for non quantization use case and if threshold is met
            if (quantizationState == null && shouldSkipBuildingVectorDataStructure(totalLiveDocs, approximateThreshold)) {
                log.debug(
                    "Skip building vector data structure for field: {}, as liveDoc: {} is less than the threshold {} during merge",
                    fieldInfo.name,
                    totalLiveDocs,
                    approximateThreshold
                );
                return;
            }
        }

        final NativeIndexWriter writer = NativeIndexWriter.getWriter(
            fieldInfo,
            segmentWriteState,
            quantizationState,
            nativeIndexBuildStrategyFactory,
            quantizedByteVectorValues
        );

        final StopWatch stopWatch = new StopWatch().start();

        writer.mergeIndex(knnVectorValuesSupplier, totalLiveDocs);

        final long time_in_millis = stopWatch.stop().totalTime().millis();
        KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.incrementBy(time_in_millis);
        log.debug("Merge took {} ms for vector field [{}]", time_in_millis, fieldInfo.getName());
    }

    public static boolean shouldSkipBuildingVectorDataStructure(final long docCount, final int approximateThreshold) {
        if (approximateThreshold < 0) {
            return true;
        }
        return docCount < approximateThreshold;
    }

    /**
     * The {@link KNNVectorValues} will be exhausted after this function run. So make sure that you are not sending the
     * vectorsValues object which you plan to use later
     */
    public static int getLiveDocs(final KNNVectorValues<?> vectorValues) throws IOException {
        // Count all the live docs as there vectorValues.totalLiveDocs() just gives the cost for the FloatVectorValues,
        // and doesn't tell the correct number of docs, if there are deleted docs in the segment. So we are counting
        // the total live docs here.
        int liveDocs = 0;
        while (vectorValues.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
            liveDocs++;
        }
        return liveDocs;
    }
}

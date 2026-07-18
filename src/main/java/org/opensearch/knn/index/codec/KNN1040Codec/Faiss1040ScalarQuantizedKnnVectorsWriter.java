/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.NonNull;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.util.IORunnable;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.knn.index.codec.nativeindex.AbstractNativeEnginesKnnVectorsWriter;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexWriter;

import java.io.IOException;
import java.util.function.Function;

/**
 * Writer for Faiss SQ vector fields. Unlike {@link org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsWriter}
 * which handles multiple fields, this writer handles exactly one field per format instance
 * (since each SQ field gets its own dedicated format via per-field routing).
 *
 * <p>Write path:
 * <ol>
 *   <li>Flat vectors are written by Lucene's SQ flat writer (.vec + .veq/.vemq files)</li>
 *   <li>HNSW graph is built by native Faiss via {@link NativeIndexWriter} (.faiss file)</li>
 * </ol>
 *
 * <p>No quantization training is needed — Lucene's flat format handles quantization internally.
 */
@Log4j2
class Faiss1040ScalarQuantizedKnnVectorsWriter extends AbstractNativeEnginesKnnVectorsWriter {
    private static final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(Faiss1040ScalarQuantizedKnnVectorsWriter.class);

    private final SegmentWriteState segmentWriteState;
    // Supplies a Lucene flat format for a given FieldInfo. Called lazily on the first addField()
    // or mergeOneField() call — this is when Lucene hands us the exact FieldInfo, so we can
    // resolve the correct ScalarEncoding from its SQ_CONFIG attribute rather than relying on a
    // pre-baked encoding at construction time. Deferring is required because at
    // Faiss1040ScalarQuantizedKnnVectorsFormat.fieldsWriter(state) time, state.fieldInfos may
    // still be null on the initial-write path (Lucene's IndexingChain calls fieldsWriter before
    // fieldInfos is populated).
    private final Function<FieldInfo, KNN1040ScalarQuantizedVectorsFormat> flatFormatResolver;
    // The lazily-constructed flat writer — non-null after the first addField/mergeOneField call.
    private FlatVectorsWriter flatVectorsWriter;
    private KNN1040ScalarQuantizedVectorsFormat resolvedFlatFormat;
    // Single field — SQ gets a dedicated format per field via BasePerFieldKnnVectorsFormat
    private FlatFieldVectorsWriter<?> fieldWriter;
    private FieldInfo fieldInfo;
    private boolean finished;
    private final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory;

    Faiss1040ScalarQuantizedKnnVectorsWriter(
        @NonNull SegmentWriteState segmentWriteState,
        @NonNull Function<FieldInfo, KNN1040ScalarQuantizedVectorsFormat> flatFormatResolver,
        @NonNull NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory
    ) {
        this.segmentWriteState = segmentWriteState;
        this.flatFormatResolver = flatFormatResolver;
        this.nativeIndexBuildStrategyFactory = nativeIndexBuildStrategyFactory;
    }

    /**
     * Resolves the flat format from the given field on first call, constructs the underlying
     * Lucene flat writer, and caches both for subsequent operations (flush, close). Called from
     * {@link #addField(FieldInfo)} and {@link #mergeOneField(FieldInfo, MergeState)}.
     */
    private FlatVectorsWriter getOrInitFlatVectorsWriter(final FieldInfo fieldInfoForResolution) throws IOException {
        if (flatVectorsWriter == null) {
            this.resolvedFlatFormat = flatFormatResolver.apply(fieldInfoForResolution);
            this.flatVectorsWriter = resolvedFlatFormat.fieldsWriter(segmentWriteState);
        }
        return flatVectorsWriter;
    }

    /**
     * Only one field is expected per format instance. Throws if called more than once.
     */
    @Override
    public KnnFieldVectorsWriter<?> addField(final FieldInfo newFieldInfo) throws IOException {
        if (this.fieldWriter != null) {
            throw new IllegalStateException(
                Faiss1040ScalarQuantizedKnnVectorsWriter.class.getSimpleName()
                    + " supports only a single field, but addField was called for ["
                    + newFieldInfo.name
                    + "] after ["
                    + this.fieldInfo.name
                    + "]"
            );
        }
        this.fieldInfo = newFieldInfo;
        this.fieldWriter = getOrInitFlatVectorsWriter(newFieldInfo).addField(newFieldInfo);
        return fieldWriter;
    }

    /**
     * Flushes flat vectors first (Lucene SQ format), then builds the native HNSW graph.
     *
     * <p>The flat writer is flushed, finished, and closed before the native build so that
     * the .vec and .veb files are fully written and file handles released. The writer then
     * opens a FlatVectorsReader to extract QuantizedByteVectorValues (via reflection on
     * Lucene internals) and passes it to the build strategy. The reader is closed after
     * the native build completes.
     */
    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        if (flatVectorsWriter == null) {
            // No field was added — nothing to flush and no native build to run.
            return;
        }
        // Flush, finish, and close the flat vectors writer so that the .vec and .veb files
        // are fully written and file handles are released.
        flatVectorsWriter.flush(maxDoc, sortMap);
        flatVectorsWriter.finish();
        IOUtils.close(flatVectorsWriter);

        if (fieldWriter == null) {
            return;
        }

        // Open a reader on the just-written flat files, extract QuantizedByteVectorValues,
        // and pass it to the build strategy. The writer owns the reader lifecycle.
        final FlatVectorsReader flatVectorsReader = openFlatVectorsReader();
        try {
            final QuantizedByteVectorValues quantizedValues = KNN1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(
                flatVectorsReader.getFloatVectorValues(fieldInfo.getName())
            );
            doFlush(
                fieldInfo,
                fieldWriter,
                fieldWriter.getVectors(),
                null,
                null,
                segmentWriteState,
                nativeIndexBuildStrategyFactory,
                quantizedValues
            );
        } finally {
            IOUtils.close(flatVectorsReader);
        }
    }

    /**
     * Merges flat vectors first, then builds the native HNSW graph for the merged segment.
     */
    @Override
    public IORunnable mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        // Setting field info
        this.fieldInfo = fieldInfo;

        // Resolve the flat format from the merged field's SQ_CONFIG on the first call. Merge
        // paths don't go through addField, so this is where lazy init happens for merges.
        final FlatVectorsWriter writer = getOrInitFlatVectorsWriter(fieldInfo);

        // Merge, finish, and close the flat writer so that files are readable.
        IORunnable mergeRunnable = writer.mergeOneField(fieldInfo, mergeState);

        if (mergeRunnable != null) mergeRunnable.run();
        writer.finish();
        IOUtils.close(writer);

        // Open a reader on the merged flat files, extract QuantizedByteVectorValues,
        // and pass it to the build strategy. The writer owns the reader lifecycle.
        final FlatVectorsReader flatVectorsReader = openFlatVectorsReader();
        try {
            final FloatVectorValues floatVectorValues = flatVectorsReader.getFloatVectorValues(fieldInfo.getName());
            if (floatVectorValues == null || floatVectorValues.size() == 0) {
                log.debug("No scalar-quantized vectors found for field [{}], skipping native build", fieldInfo.getName());
                return null;
            }
            final QuantizedByteVectorValues quantizedValues = KNN1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(
                floatVectorValues
            );
            doMergeOneField(fieldInfo, mergeState, null, null, segmentWriteState, nativeIndexBuildStrategyFactory, quantizedValues);
        } finally {
            IOUtils.close(flatVectorsReader);
        }
        return null;
    }

    @Override
    public void finish() throws IOException {
        if (finished) {
            throw new IllegalStateException(Faiss1040ScalarQuantizedKnnVectorsWriter.class.getSimpleName() + " is already finished");
        }
        finished = true;
        // flatVectorsWriter.finish() and close() are already called in flush/mergeOneField
        // before the native build. No additional finalization needed here.
    }

    @Override
    public void close() throws IOException {
        // flatVectorsWriter is already closed in flush/mergeOneField.
        // IOUtils.close is safe to call on an already-closed resource.
        IOUtils.close(flatVectorsWriter);
    }

    @Override
    public long ramBytesUsed() {
        return SHALLOW_SIZE + (flatVectorsWriter != null ? flatVectorsWriter.ramBytesUsed() : 0) + (fieldWriter != null
            ? fieldWriter.ramBytesUsed()
            : 0);
    }

    /**
     * Opens a FlatVectorsReader scoped to this single field from the already-written flat files.
     * Uses the resolved flat format from the write phase to ensure encoding matches.
     */
    private FlatVectorsReader openFlatVectorsReader() throws IOException {
        final SegmentReadState readState = new SegmentReadState(
            segmentWriteState.directory,
            segmentWriteState.segmentInfo,
            new FieldInfos(new FieldInfo[] { fieldInfo }),
            segmentWriteState.context,
            segmentWriteState.segmentSuffix
        );
        return resolvedFlatFormat.fieldsReader(readState);
    }
}

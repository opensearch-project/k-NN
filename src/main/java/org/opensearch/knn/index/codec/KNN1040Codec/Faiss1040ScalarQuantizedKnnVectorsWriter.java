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
import org.apache.lucene.codecs.lucene104.QuantizedByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.util.IOFunction;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.knn.index.codec.nativeindex.AbstractNativeEnginesKnnVectorsWriter;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;

import java.io.IOException;
import java.lang.reflect.Field;

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
class Faiss1040ScalarQuantizedKnnVectorsWriter extends AbstractNativeEnginesKnnVectorsWriter {
    private static final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(Faiss1040ScalarQuantizedKnnVectorsWriter.class);

    private final SegmentWriteState segmentWriteState;
    private final FlatVectorsWriter flatVectorsWriter;
    // Single field — BBQ gets a dedicated format per field via BasePerFieldKnnVectorsFormat
    private FlatFieldVectorsWriter<?> fieldWriter;
    private FieldInfo fieldInfo;
    private boolean finished;
    private final IOFunction<SegmentReadState, FlatVectorsReader> quantizedFlatVectorsReaderSupplier;

    Faiss1040ScalarQuantizedKnnVectorsWriter(
        @NonNull SegmentWriteState segmentWriteState,
        @NonNull FlatVectorsWriter flatVectorsWriter,
        @NonNull IOFunction<SegmentReadState, FlatVectorsReader> quantizedFlatVectorsReaderSupplier
    ) {
        this.segmentWriteState = segmentWriteState;
        this.flatVectorsWriter = flatVectorsWriter;
        this.quantizedFlatVectorsReaderSupplier = quantizedFlatVectorsReaderSupplier;
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
        this.fieldWriter = flatVectorsWriter.addField(newFieldInfo);
        return fieldWriter;
    }

    /**
     * Flushes flat vectors first (Lucene BBQ format), then builds the native HNSW graph.
     *
     * <p>The flat writer is flushed, finished, and closed before the native build so that
     * the .vec and .veb files are fully written and file handles released. The writer then
     * opens a FlatVectorsReader to extract QuantizedByteVectorValues (via reflection on
     * Lucene internals) and passes it to the build strategy. The reader is closed after
     * the native build completes.
     */
    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
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
            final QuantizedByteVectorValues quantizedValues = extractQuantizedByteVectorValues(flatVectorsReader);
            doFlush(
                fieldInfo,
                fieldWriter,
                fieldWriter.getVectors(),
                null,
                null,
                segmentWriteState,
                new NativeIndexBuildStrategyFactory(),
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
    public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        // Setting field info
        this.fieldInfo = fieldInfo;

        // Merge, finish, and close the flat writer so that files are readable.
        flatVectorsWriter.mergeOneField(fieldInfo, mergeState);
        flatVectorsWriter.finish();
        IOUtils.close(flatVectorsWriter);

        // Open a reader on the merged flat files, extract QuantizedByteVectorValues,
        // and pass it to the build strategy. The writer owns the reader lifecycle.
        final FlatVectorsReader flatVectorsReader = openFlatVectorsReader();
        try {
            final QuantizedByteVectorValues quantizedValues = extractQuantizedByteVectorValues(flatVectorsReader);
            doMergeOneField(fieldInfo, mergeState, null, null, segmentWriteState, new NativeIndexBuildStrategyFactory(), quantizedValues);
        } finally {
            IOUtils.close(flatVectorsReader);
        }
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
        return SHALLOW_SIZE + flatVectorsWriter.ramBytesUsed() + (fieldWriter != null ? fieldWriter.ramBytesUsed() : 0);
    }

    /**
     * Opens a FlatVectorsReader scoped to this single field from the already-written flat files.
     */
    private FlatVectorsReader openFlatVectorsReader() throws IOException {
        final SegmentReadState readState = new SegmentReadState(
            segmentWriteState.directory,
            segmentWriteState.segmentInfo,
            new FieldInfos(new FieldInfo[] { fieldInfo }),
            segmentWriteState.context,
            fieldInfo.getName()
        );
        return quantizedFlatVectorsReaderSupplier.apply(readState);
    }

    /**
     * Extracts QuantizedByteVectorValues from a FlatVectorsReader via reflection.
     *
     * <p>The FlatVectorsReader.getFloatVectorValues() returns a BinarizedVectorValues instance
     * that wraps both the quantized binary codes and their correction factors. The underlying
     * QuantizedByteVectorValues is accessed via the private "quantizedVectorValues" field.
     *
     * <p>This reflection is necessary because Lucene does not expose a public API to access
     * the quantized byte vector values directly from the reader.
     */
    private QuantizedByteVectorValues extractQuantizedByteVectorValues(final FlatVectorsReader reader) throws IOException {
        try {
            final FloatVectorValues floatVectorValues = reader.getFloatVectorValues(fieldInfo.getName());
            final Field f = floatVectorValues.getClass().getDeclaredField("quantizedVectorValues");
            f.setAccessible(true);
            return (QuantizedByteVectorValues) f.get(floatVectorValues);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new IOException(
                "Failed to extract QuantizedByteVectorValues from FlatVectorsReader for field ["
                    + fieldInfo.getName()
                    + "]. This may indicate an incompatible Lucene version.",
                e
            );
        }
    }
}

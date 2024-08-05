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

import lombok.RequiredArgsConstructor;
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
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexWriter;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.opensearch.knn.common.FieldInfoExtractor.extractVectorDataType;

/**
 * A KNNVectorsWriter class for writing the vector data strcutures and flat vectors for Native Engines.
 */
@Log4j2
@RequiredArgsConstructor
public class NativeEngines990KnnVectorsWriter extends KnnVectorsWriter {
    private static final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(NativeEngines990KnnVectorsWriter.class);
    private final SegmentWriteState segmentWriteState;
    private final FlatVectorsWriter flatVectorsWriter;
    private final List<NativeEngineFieldVectorsWriter<?>> fields = new ArrayList<>();
    private boolean finished;

    /**
     * Add new field for indexing.
     * In Lucene, we use single file for all the vector fields so here we need to see how we are going to make things
     * work.
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
        // simply write data in the flat file
        flatVectorsWriter.flush(maxDoc, sortMap);
        for (final NativeEngineFieldVectorsWriter<?> field : fields) {
            final VectorDataType vectorDataType = extractVectorDataType(field.getFieldInfo());
            final KNNVectorValues<?> knnVectorValues = KNNVectorValuesFactory.getVectorValues(
                vectorDataType,
                field.getDocsWithField(),
                field.getVectors()
            );

            // TODO: Extract quantization state here
            NativeIndexWriter.getWriter(field.getFieldInfo(), segmentWriteState).flushIndex(knnVectorValues);
        }
    }

    @Override
    public void mergeOneField(final FieldInfo fieldInfo, final MergeState mergeState) throws IOException {
        // This will ensure that we are merging the FlatIndex during force merge.
        flatVectorsWriter.mergeOneField(fieldInfo, mergeState);

        // For merge, pick values from flat vector and reindex again. This will use the flush operation to create graphs
        final VectorDataType vectorDataType = extractVectorDataType(fieldInfo);
        final KNNVectorValues<?> knnVectorValues;
        switch (fieldInfo.getVectorEncoding()) {
            case FLOAT32:
                final FloatVectorValues mergedFloats = MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
                knnVectorValues = KNNVectorValuesFactory.getVectorValues(vectorDataType, mergedFloats);
                break;
            case BYTE:
                final ByteVectorValues mergedBytes = MergedVectorValues.mergeByteVectorValues(fieldInfo, mergeState);
                knnVectorValues = KNNVectorValuesFactory.getVectorValues(vectorDataType, mergedBytes);
                break;
            default:
                throw new IllegalStateException("Unsupported vector encoding [" + fieldInfo.getVectorEncoding() + "]");
        }

        // TODO: Extract Quantization state here
        NativeIndexWriter.getWriter(fieldInfo, segmentWriteState).mergeIndex(knnVectorValues);
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

}

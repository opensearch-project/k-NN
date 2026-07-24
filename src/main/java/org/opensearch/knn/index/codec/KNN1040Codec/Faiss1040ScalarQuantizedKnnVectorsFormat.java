/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import static org.opensearch.knn.common.KNNConstants.SQ_CONFIG;

/**
 * Dedicated format for Faiss SQ vector fields.
 *
 * <p>Uses {@link KNN1040ScalarQuantizedVectorsFormat} for flat vector storage (.vec/.veq files),
 * while HNSW graph construction is delegated to the native Faiss engine (.faiss files). The
 * {@link ScalarEncoding} is resolved <b>per-field at write time</b> from the segment's
 * {@link FieldInfo} attributes ({@code SQ_CONFIG}, populated by {@code EngineFieldMapper}) —
 * <b>not</b> from a value baked into the format instance at construction time.
 *
 * <p>This matters because Lucene's SPI machinery ({@link KnnVectorsFormat#forName(String)},
 * invoked by {@code PerFieldKnnVectorsFormat.FieldsReader}) reopens codecs via the mandatory
 * no-arg constructor whenever segments are opened on a new reader (index open, shard relocation,
 * node bootup). If encoding were carried on the instance, an SPI-instantiated format would hold
 * a constructor default (previously {@code SINGLE_BIT_QUERY_NIBBLE}) and any writer produced by
 * it would silently miswrite non-1-bit fields as 1-bit. Resolving encoding per-field from
 * {@code SQ_CONFIG} makes the format instance encoding-agnostic.
 *
 * <p>On the <b>read path</b>, {@link org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat#fieldsReader} never
 * threads its format-instance encoding through to
 * {@code Lucene104ScalarQuantizedVectorsReader} — the reader resolves per-field encoding by
 * reading the wire number from each field's {@code .vemq} meta sidecar (scoped by
 * {@code SegmentReadState.segmentSuffix} at the Lucene layer). So the encoding we pass to
 * {@link KNN1040ScalarQuantizedVectorsFormat} on the read path is unused — we intentionally use
 * a default-encoding instance rather than pretend to resolve it.
 *
 * @see Faiss1040ScalarQuantizedKnnVectorsWriter
 * @see Faiss1040ScalarQuantizedKnnVectorsReader
 */
@Log4j2
public class Faiss1040ScalarQuantizedKnnVectorsFormat extends KnnVectorsFormat {

    private static final String FORMAT_NAME = "Faiss1040ScalarQuantizedKnnVectorsFormat";

    // KNN1040ScalarQuantizedVectorsFormat is stateless per encoding, so we cache one instance per
    // encoding and share it across all format instances.
    private static final Map<ScalarEncoding, KNN1040ScalarQuantizedVectorsFormat> FLAT_FORMAT_CACHE = new ConcurrentHashMap<>();

    private final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory;

    private static KNN1040ScalarQuantizedVectorsFormat flatFormatFor(final ScalarEncoding encoding) {
        return FLAT_FORMAT_CACHE.computeIfAbsent(encoding, KNN1040ScalarQuantizedVectorsFormat::new);
    }

    /**
     * Returns a fresh {@link KNN1040ScalarQuantizedVectorsFormat} with the default encoding. Used
     * on the read path, where the encoding is unused downstream (see {@link #fieldsReader}).
     */
    private static KNN1040ScalarQuantizedVectorsFormat flatFormatFor() {
        return new KNN1040ScalarQuantizedVectorsFormat();
    }

    public Faiss1040ScalarQuantizedKnnVectorsFormat() {
        this(new NativeIndexBuildStrategyFactory());
    }

    public Faiss1040ScalarQuantizedKnnVectorsFormat(final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory) {
        super(FORMAT_NAME);
        this.nativeIndexBuildStrategyFactory = nativeIndexBuildStrategyFactory;
    }

    /**
     * Resolves the {@link ScalarEncoding} for {@code fieldInfo} by reading the field's
     * {@code SQ_CONFIG} attribute (set by {@code EngineFieldMapper}). Used only by the write
     * path — the read path does not consume format-instance encoding (see class javadoc).
     *
     * <p>Contract: every field routed to this format is expected to be an SQ field carrying
     * {@code SQ_CONFIG} with a positive {@code bits} value (1, 2, or 4).
     * {@code EngineFieldMapper} sets this attribute unconditionally on the mapping side, and
     * Lucene's {@code PerFieldKnnVectorsFormat} only routes a field to this format when the
     * mapper decided this format applies. So reaching this method with a bits {@code <= 0}
     * field indicates an upstream invariant violation — we fail loud rather than silently
     * defaulting to 1-bit (the silent-default was exactly the class of bug this refactor was
     * written to eliminate).
     */
    private static ScalarEncoding resolveEncodingForField(final FieldInfo fieldInfo) {
        final int bits = FieldInfoExtractor.extractSQConfig(fieldInfo).getBits();
        if (bits <= 0) {
            throw new IllegalStateException(
                String.format(
                    Locale.ROOT,
                    "%s: SQ field [%s] has no bits set in its %s attribute; bits must be populated for every SQ field. Supported bits: %s",
                    FORMAT_NAME,
                    fieldInfo.getName(),
                    SQ_CONFIG,
                    ScalarEncodingResolver.supportedDocBitsString()
                )
            );
        }
        return ScalarEncodingResolver.forDocBits(bits);
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        // Encoding is resolved LAZILY inside the wrapper writer, on the first addField() or
        // mergeOneField() call (that's when Lucene hands us the exact FieldInfo). We cannot
        // resolve here because state.fieldInfos may be null on the initial-write path — Lucene's
        // IndexingChain calls fieldsWriter() before FieldInfos is populated.
        return new Faiss1040ScalarQuantizedKnnVectorsWriter(
            state,
            fieldInfo -> flatFormatFor(resolveEncodingForField(fieldInfo)),
            nativeIndexBuildStrategyFactory
        );
    }

    /**
     * Wraps the Lucene flat vectors reader with {@link Faiss1040ScalarQuantizedFlatVectorsReader} so that
     * the {@link org.apache.lucene.index.FloatVectorValues} returned by the reader implement
     * {@link org.apache.lucene.codecs.lucene95.HasIndexSlice}. This is required because Lucene's
     * HNSW traversal expects all vector values to expose an {@link org.apache.lucene.store.IndexInput},
     * but Lucene's {@code ScalarQuantizedVectorValues} does not implement that interface.
     *
     * <p><b>Encoding is not resolved on the read path.</b> {@link org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat#fieldsReader}
     * ({@code Lucene104ScalarQuantizedVectorsFormat.java:133}) constructs {@code Lucene104ScalarQuantizedVectorsReader}
     * <i>without</i> threading in the format instance's encoding — the reader instead resolves the encoding for
     * each field by reading the wire number from the field's own {@code .vemq} meta sidecar
     * ({@code Lucene104ScalarQuantizedVectorsReader.FieldEntry.create}, {@code input.readVInt()} →
     * {@code ScalarEncoding.fromWireNumber}).
     *
     * <p>Therefore, passing a specific encoding to {@link KNN1040ScalarQuantizedVectorsFormat} here would
     * be dead code — the value is unused downstream. We deliberately use the default no-arg constructor
     * to avoid implying a per-field resolution that isn't happening.
     */
    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new Faiss1040ScalarQuantizedKnnVectorsReader(
            state,
            new Faiss1040ScalarQuantizedFlatVectorsReader(flatFormatFor().fieldsReader(state))
        );
    }

    /**
     * Uses Faiss max dimension since the HNSW graph is built by native Faiss.
     */
    @Override
    public int getMaxDimensions(String fieldName) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.FAISS);
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName() + "(name=" + this.getClass().getSimpleName() + ")";
    }
}

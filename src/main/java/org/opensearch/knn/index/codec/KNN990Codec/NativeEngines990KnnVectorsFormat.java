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
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;

/**
 * This is a Vector format that will be used for Native engines like Faiss and Nmslib for reading and writing vector
 * related data structures.
 */
@Log4j2
public class NativeEngines990KnnVectorsFormat extends KnnVectorsFormat {
    /** The format for storing, reading, merging vectors on disk */
    private static final FlatVectorsFormat flatVectorsFormat = new Lucene99FlatVectorsFormat(
        FlatVectorScorerUtil.getLucene99FlatVectorsScorer()
    );
    private static final String FORMAT_NAME = "NativeEngines990KnnVectorsFormat";
    private final int approximateThreshold;
    private final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory;

    public NativeEngines990KnnVectorsFormat() {
        this(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE);
    }

    public NativeEngines990KnnVectorsFormat(int approximateThreshold) {
        this(approximateThreshold, new NativeIndexBuildStrategyFactory());
    }

    public NativeEngines990KnnVectorsFormat(
        int approximateThreshold,
        final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory
    ) {
        super(FORMAT_NAME);
        this.approximateThreshold = approximateThreshold;
        this.nativeIndexBuildStrategyFactory = nativeIndexBuildStrategyFactory;
    }

    /**
     * Returns a {@link KnnVectorsWriter} to write the vectors to the index.
     *
     * @param state {@link SegmentWriteState}
     */
    @Override
    public KnnVectorsWriter fieldsWriter(final SegmentWriteState state) throws IOException {
        return new NativeEngines990KnnVectorsWriter(
            state,
            flatVectorsFormat.fieldsWriter(state),
            approximateThreshold,
            nativeIndexBuildStrategyFactory
        );
    }

    /**
     * Returns a {@link KnnVectorsReader} to read the vectors from the index.
     *
     * @param state {@link SegmentReadState}
     */
    @Override
    public KnnVectorsReader fieldsReader(final SegmentReadState state) throws IOException {
        return new NativeEngines990KnnVectorsReader(state, flatVectorsFormat.fieldsReader(state));
    }

    /**
     * Returns the maximum number of vector dimensions supported by this codec for the given field
     * name
     *
     * <p>Codecs implement this method to specify the maximum number of dimensions they support.
     *
     * Even though this codec is used for both Nmslib and Faiss, but we are usin Faiss Engine here since Nmslib is
     * deprecated
     *
     * @param fieldName the field name
     * @return the maximum number of vector dimensions.
     */
    @Override
    public int getMaxDimensions(final String fieldName) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.FAISS);
    }

    @Override
    public String toString() {
        return "NativeEngines99KnnVectorsFormat(name="
            + this.getClass().getSimpleName()
            + ", flatVectorsFormat="
            + flatVectorsFormat
            + ", approximateThreshold="
            + approximateThreshold
            + ")";
    }
}

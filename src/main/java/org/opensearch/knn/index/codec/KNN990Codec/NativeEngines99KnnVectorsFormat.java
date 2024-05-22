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

import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

import java.io.IOException;

public class NativeEngines99KnnVectorsFormat extends KnnVectorsFormat {

    /** The format for storing, reading, merging vectors on disk */
    private static final FlatVectorsFormat flatVectorsFormat = new Lucene99FlatVectorsFormat(new DefaultFlatVectorScorer());

    /**
     * Sole constructor
     *
     */
    public NativeEngines99KnnVectorsFormat() {
        super("NativeEngines99KnnVectorsFormat");
    }

    /**
     * Returns a {@link KnnVectorsWriter} to write the vectors to the index.
     *
     * @param state
     */
    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new NativeEnginesKNNVectorsWriter(state, flatVectorsFormat.fieldsWriter(state));
    }

    /**
     * Returns a {@link KnnVectorsReader} to read the vectors from the index.
     *
     * @param state
     */
    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new NativeEnginesKNNVectorsReader(state, flatVectorsFormat.fieldsReader(state));
    }

    @Override
    public String toString() {
        return "NativeEngines99KnnVectorsFormat(name=NativeEngines99KnnVectorsFormat, flatVectorsFormat=" + flatVectorsFormat + ")";
    }

}

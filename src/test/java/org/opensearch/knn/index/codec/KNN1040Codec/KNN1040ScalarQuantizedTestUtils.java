/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

final class KNN1040ScalarQuantizedTestUtils {

    static final String FIELD_NAME = "vector";
    static final int DIMENSION = 128;
    static final int NUM_VECTORS = 50;

    private KNN1040ScalarQuantizedTestUtils() {}

    // Uses single-segment flush path (not merge). Sufficient for verifying scorer/reader types since
    // those are determined by the flat format, not by how the segment was created.
    // The returned SegmentReadState references the passed-in directory; caller must ensure
    // the directory outlives any readers opened from this state.
    static SegmentReadState writeQuantizedVectors(MMapDirectory dir, KNN1040ScalarQuantizedVectorsFormat format, Random random)
        throws Exception {
        final FieldInfo fieldInfo = new FieldInfo(
            FIELD_NAME,
            0,
            false,
            false,
            false,
            IndexOptions.NONE,
            DocValuesType.NONE,
            DocValuesSkipIndexType.NONE,
            -1,
            Map.of(),
            0,
            0,
            0,
            DIMENSION,
            VectorEncoding.FLOAT32,
            VectorSimilarityFunction.EUCLIDEAN,
            false,
            false
        );
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });

        final SegmentInfo segmentInfo = new SegmentInfo(
            dir,
            Version.LATEST,
            Version.LATEST,
            "_0",
            NUM_VECTORS,
            false,
            false,
            null,
            Collections.emptyMap(),
            StringHelper.randomId(),
            new HashMap<>(),
            null
        );
        final SegmentWriteState writeState = new SegmentWriteState(
            InfoStream.NO_OUTPUT,
            dir,
            segmentInfo,
            fieldInfos,
            null,
            IOContext.DEFAULT
        );

        try (FlatVectorsWriter writer = format.fieldsWriter(writeState)) {
            @SuppressWarnings("unchecked")
            FlatFieldVectorsWriter<float[]> fieldWriter = (FlatFieldVectorsWriter<float[]>) writer.addField(fieldInfo);

            for (int i = 0; i < NUM_VECTORS; i++) {
                fieldWriter.addValue(i, randomVector(DIMENSION, random));
            }

            // null sortMap means unsorted segment — standard for non-index-time-sorted segments
            writer.flush(NUM_VECTORS, null);
            writer.finish();
        }

        return new SegmentReadState(dir, segmentInfo, fieldInfos, IOContext.DEFAULT);
    }

    static float[] randomVector(int dimension, Random random) {
        float[] v = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            v[i] = random.nextFloat() * 2 - 1;
        }
        return v;
    }
}

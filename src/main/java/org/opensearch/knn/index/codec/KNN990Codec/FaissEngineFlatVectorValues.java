/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.SpaceType;

import java.io.IOException;
import java.util.Arrays;

import static org.opensearch.knn.index.codec.KNN990Codec.FaissEngineFlatKnnVectorsReader.FLOAT_SIZE;
import static org.opensearch.knn.index.codec.KNN990Codec.FaissEngineFlatKnnVectorsReader.SIZET_SIZE;

public class FaissEngineFlatVectorValues extends FloatVectorValues {
    private static final int BUCKET_VECTORS = 64; //every time read only bucket size vectors.
    protected FaissEngineFlatKnnVectorsReader.MetaInfo metaInfo;
    protected final IndexInput slice;
    protected final VectorSimilarityFunction similarityFunction;
    protected final FlatVectorsScorer flatVectorsScorer;
    protected final float[] value;
    protected final long[] ids;
    protected final float[] buf;
    protected int docId = -1;
    protected int ord = -1;

    public FaissEngineFlatVectorValues(FaissEngineFlatKnnVectorsReader.MetaInfo metaInfo, IndexInput input) throws IOException {
        this.metaInfo = metaInfo;
        this.slice = input.clone();
        this.similarityFunction = getVectorSimilarityFunction(metaInfo.metricType).getVectorSimilarityFunction();
        this.flatVectorsScorer = FlatVectorScorerUtil.getLucene99FlatVectorsScorer();
        this.value = new float[(int) (metaInfo.d * metaInfo.ntotal)];
        this.ids= new long[(int) metaInfo.ntotal];
        this.buf = new float[metaInfo.d];
        readIds();
    }

    protected void readIds() throws IOException {
        slice.seek(metaInfo.idSeek);
        long size = slice.readLong();
        assert size == metaInfo.ntotal;
        slice.readLongs(ids, 0, (int) metaInfo.ntotal);
    }

    protected void readBucketVectors() throws IOException {
        assert ord >= 0;
        assert ord <= metaInfo.ntotal;
        int bucketIndex = ord / BUCKET_VECTORS;
        slice.seek(metaInfo.vectorSeek + SIZET_SIZE + bucketIndex * BUCKET_VECTORS * FLOAT_SIZE * metaInfo.d);

        for (int i = 0, o = ord;
                i < BUCKET_VECTORS && o < metaInfo.ntotal;
                i++, o++) {
            slice.readFloats(value, i * metaInfo.d, metaInfo.d);
        }
    }
//    public void readInfo() throws IOException {
//        slice.seek(metaInfo.idSeek);
//        long size = slice.readLong();
//        assert size == metaInfo.ntotal;
//        slice.readLongs(ids, 0, (int) metaInfo.ntotal);
//
//        slice.seek(metaInfo.vectorSeek);
//        size = slice.readLong();
//        for(int i = 0; i < metaInfo.ntotal; i++) {
//            slice.readFloats(value, i * metaInfo.d, metaInfo.d);
//        }
//    }

    @Override
    public int dimension() {
        return metaInfo.d;
    }

    @Override
    public int size() {
        return (int) metaInfo.ntotal;
    }

    @Override
    public float[] vectorValue() throws IOException {
        if(ord % BUCKET_VECTORS == 0) {
            readBucketVectors();
        }
        int bucketOrder = ord % BUCKET_VECTORS;

        System.arraycopy(value, bucketOrder * metaInfo.d, buf, 0, metaInfo.d);
        return buf;
    }

    @Override
    public VectorScorer scorer(float[] floats) throws IOException {
        //TODO
        return null;
    }

    @Override
    public int docID() {
        return docId;
    }

    @Override
    public int nextDoc() throws IOException {
        return advance(docId + 1);
    }

    @Override
    public int advance(int target) throws IOException {
        ord = Arrays.binarySearch(ids, ord + 1, ids.length, target);
        if (ord < 0) {
            ord = -(ord + 1);
        }
        assert ord <= ids.length;
        if (ord == ids.length) {
            docId = NO_MORE_DOCS;
        } else {
            docId = (int) ids[ord];
        }
        return docId;
    }

    KNNVectorSimilarityFunction getVectorSimilarityFunction(int metricType) {
        // Ref from jni/external/faiss/c_api/Index_c.h
        switch (metricType) {
            case 0:
                return SpaceType.INNER_PRODUCT.getKnnVectorSimilarityFunction();
            case 1:
                return SpaceType.L2.getKnnVectorSimilarityFunction();
            case 2:
                return SpaceType.L1.getKnnVectorSimilarityFunction();
            case 3:
                return SpaceType.LINF.getKnnVectorSimilarityFunction();
            default:
                return SpaceType.L2.getKnnVectorSimilarityFunction();
        }
    }
}

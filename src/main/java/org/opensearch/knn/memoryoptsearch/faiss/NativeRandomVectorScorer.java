/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.NonNull;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.jni.SimdVectorComputeService;

import java.io.IOException;

public class NativeRandomVectorScorer implements RandomVectorScorer {
    @NonNull
    private final KnnVectorValues knnVectorValues;
    private long[] addressAndSize;
    private int maxOrd;
    private int nativeFunctionTypeOrd;

    public NativeRandomVectorScorer(
        final float[] query,
        final KnnVectorValues knnVectorValues,
        final MMapVectorValues mmapVectorValues,
        final SimdVectorComputeService.SimilarityFunctionType similarityFunctionType
    ) {
        this.knnVectorValues = knnVectorValues;
        this.addressAndSize = mmapVectorValues.getAddressAndSize();
        this.maxOrd = knnVectorValues.size();
        this.nativeFunctionTypeOrd = similarityFunctionType.ordinal();
        SimdVectorComputeService.saveSearchContext(query, addressAndSize, nativeFunctionTypeOrd);
    }

    @Override
    public void bulkScore(final int[] internalVectorIds, final float[] scores, final int numVectors) {
        SimdVectorComputeService.bulkDistanceCalculation(internalVectorIds, scores, numVectors);
    }

    @Override
    public float score(final int internalVectorId) throws IOException {
        return SimdVectorComputeService.scoreSingleVector(internalVectorId);
    }

    @Override
    public int maxOrd() {
        return maxOrd;
    }

    @Override
    public int ordToDoc(int ord) {
        return knnVectorValues.ordToDoc(ord);
    }

    @Override
    public Bits getAcceptOrds(Bits acceptDocs) {
        return knnVectorValues.getAcceptOrds(acceptDocs);
    }
}

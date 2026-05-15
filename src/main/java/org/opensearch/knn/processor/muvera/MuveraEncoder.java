/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.processor.muvera;

import org.apache.lucene.util.VectorUtil;

import java.util.Arrays;
import java.util.Random;

/**
 * MUVERA (Multi-Vector Retrieval via Fixed Dimensional Encodings) encoder.
 *
 * <p>Converts variable-length multi-vector embeddings (e.g. ColBERT/ColPali token
 * embeddings) into a fixed-size single vector whose dot product approximates MaxSim
 * scoring. The output FDE dimension is {@code rReps * 2^kSim * dimProj}.
 *
 * <p>Document encoding ({@link #processDocument(float[][])}) normalizes cluster
 * centers by count and fills empty clusters via Hamming-nearest neighbor.
 * Query encoding ({@link #processQuery(float[][])}) sums tokens per cluster with
 * no normalization or empty-cluster filling.
 *
 * <p>The random seed must be identical between index time and query time to produce
 * compatible encodings.
 *
 * <p>Hot path internals use {@code float[]} arrays and call into Lucene's
 * {@link VectorUtil#dotProduct(float[], float[])} so the JVM can emit SIMD/FMA
 * instructions for both SimHash partitioning and the random projection step.
 */
public class MuveraEncoder {

    /**
     * Maximum allowed FDE dimension. Matches the k-NN engine max dimension limit
     * (16,000) so the FDE vector can be indexed in any supported engine.
     */
    static final int MAX_FDE_DIMENSION = 16_000;

    private final int dim;
    private final int kSim;
    private final int dimProj;
    private final int rReps;
    private final int numPartitions;

    /**
     * SimHash hyperplanes per repetition. Each plane is a contiguous {@code float[dim]}
     * stored at {@code simhashVectors[r][k]} so it can be passed directly to
     * {@link VectorUtil#dotProduct} without slicing or copying.
     */
    private final float[][][] simhashVectors;

    /**
     * Random projection matrices per repetition. Each row is a contiguous {@code float[dim]}
     * stored at {@code dimReductionProjections[r][j]} so it can be passed directly to
     * {@link VectorUtil#dotProduct} without slicing or copying.
     */
    private final float[][][] dimReductionProjections;

    /**
     * Builds a MUVERA encoder with deterministic random projections.
     *
     * @param dim     input vector dimension
     * @param kSim    number of SimHash bits ({@code 2^kSim} partitions per repetition)
     * @param dimProj output dimension per partition after random projection
     * @param rReps   number of independent repetitions
     * @param seed    seed for the {@link Random} used to generate hyperplanes and projections
     * @throws IllegalArgumentException if any dimension is non-positive or the resulting
     *                                  FDE size exceeds {@link #MAX_FDE_DIMENSION}
     */
    public MuveraEncoder(int dim, int kSim, int dimProj, int rReps, long seed) {
        if (dim <= 0) {
            throw new IllegalArgumentException("dim must be positive, got: " + dim);
        }
        if (kSim < 0) {
            throw new IllegalArgumentException("k_sim must be non-negative, got: " + kSim);
        }
        if (dimProj <= 0) {
            throw new IllegalArgumentException("dim_proj must be positive, got: " + dimProj);
        }
        if (rReps <= 0) {
            throw new IllegalArgumentException("r_reps must be positive, got: " + rReps);
        }

        // FDE dimension = rReps * 2^kSim * dimProj must fit the engine cap.
        long fdeDimension = (long) rReps * (1L << kSim) * dimProj;
        if (fdeDimension > MAX_FDE_DIMENSION) {
            throw new IllegalArgumentException(
                "MUVERA parameters produce an FDE dimension of ["
                    + fdeDimension
                    + "] which exceeds the maximum allowed dimension of ["
                    + MAX_FDE_DIMENSION
                    + "]. Reduce r_reps, k_sim, or dim_proj. (r_reps="
                    + rReps
                    + " * 2^k_sim="
                    + (1L << kSim)
                    + " * dim_proj="
                    + dimProj
                    + ")"
            );
        }

        this.dim = dim;
        this.kSim = kSim;
        this.dimProj = dimProj;
        this.rReps = rReps;
        this.numPartitions = 1 << kSim;

        Random rng = new Random(seed);

        // Each SimHash plane is its own float[dim] so the dot-product call is allocation-free.
        simhashVectors = new float[rReps][kSim][dim];
        for (int r = 0; r < rReps; r++) {
            for (int k = 0; k < kSim; k++) {
                float[] plane = simhashVectors[r][k];
                for (int d = 0; d < dim; d++) {
                    plane[d] = (float) rng.nextGaussian();
                }
            }
        }

        // Each projection row is its own float[dim] with entries in {-1, +1}.
        dimReductionProjections = new float[rReps][dimProj][dim];
        for (int r = 0; r < rReps; r++) {
            for (int j = 0; j < dimProj; j++) {
                float[] row = dimReductionProjections[r][j];
                for (int d = 0; d < dim; d++) {
                    row[d] = rng.nextBoolean() ? 1.0f : -1.0f;
                }
            }
        }
    }

    /**
     * Returns the output FDE dimension: {@code rReps * 2^kSim * dimProj}.
     *
     * @return the FDE vector length produced by {@link #processDocument(float[][])} and
     *         {@link #processQuery(float[][])}
     */
    public int getEmbeddingSize() {
        return rReps * numPartitions * dimProj;
    }

    /**
     * Computes the SimHash partition id (0..{@code numPartitions}-1) for a single token
     * vector under the {@code repIndex}-th set of hyperplanes.
     */
    private int getClusterId(float[] vector, int repIndex) {
        int clusterId = 0;
        float[][] planes = simhashVectors[repIndex];
        for (int k = 0; k < kSim; k++) {
            if (VectorUtil.dotProduct(vector, planes[k]) > 0f) {
                clusterId |= (1 << k);
            }
        }
        return clusterId;
    }

    private static int hammingDistance(int a, int b) {
        return Integer.bitCount(a ^ b);
    }

    /**
     * Encode document multi-vectors into a single FDE vector.
     *
     * <p>Cluster centers are normalized by token count, and empty clusters are filled
     * from the Hamming-nearest non-empty cluster.
     *
     * @param vectors multi-vector input, shape {@code [numVectors][dim]}
     * @return FDE vector of length {@link #getEmbeddingSize()}
     */
    public float[] processDocument(float[][] vectors) {
        return process(vectors, true, true);
    }

    /**
     * Encode query multi-vectors into a single FDE vector.
     *
     * <p>Cluster centers are summed (no normalization) and empty clusters are left as zeros.
     *
     * @param vectors multi-vector input, shape {@code [numVectors][dim]}
     * @return FDE vector of length {@link #getEmbeddingSize()}
     */
    public float[] processQuery(float[][] vectors) {
        return process(vectors, false, false);
    }

    /**
     * Core FDE encoding shared by document and query paths.
     *
     * @param vectors   multi-vector input, shape {@code [numVectors][dim]}
     * @param fillEmpty if true, fill empty clusters from Hamming-nearest non-empty cluster (document mode)
     * @param normalize if true, normalize cluster centers by count (document mode)
     * @return FDE vector of length {@link #getEmbeddingSize()}
     */
    private float[] process(float[][] vectors, boolean fillEmpty, boolean normalize) {
        int nVectors = vectors.length;
        float[] output = new float[getEmbeddingSize()];
        int outOffset = 0;
        float scale = (float) (1.0 / Math.sqrt(dimProj));

        // Pre-allocate reusable buffers so the per-repetition loop is allocation-free.
        // clusterVecIndices uses a flat int[] per cluster (sized for the worst case where
        // all tokens land in one cluster) plus per-cluster counts. This avoids
        // List<List<Integer>> boxing on the hot path.
        float[][] centers = new float[numPartitions][dim];
        int[] counts = new int[numPartitions];
        int[][] clusterVecIndices = new int[numPartitions][nVectors];

        for (int r = 0; r < rReps; r++) {
            // Reset reusable buffers for this repetition.
            for (int c = 0; c < numPartitions; c++) {
                Arrays.fill(centers[c], 0f);
                counts[c] = 0;
            }

            // Assign vectors to clusters and accumulate sums.
            for (int v = 0; v < nVectors; v++) {
                float[] tokenVec = vectors[v];
                int cid = getClusterId(tokenVec, r);
                clusterVecIndices[cid][counts[cid]] = v;
                counts[cid]++;
                float[] centerVec = centers[cid];
                for (int d = 0; d < dim; d++) {
                    centerVec[d] += tokenVec[d];
                }
            }

            // Normalize by count (document mode only).
            if (normalize) {
                for (int c = 0; c < numPartitions; c++) {
                    int count = counts[c];
                    if (count > 1) {
                        float inv = 1f / count;
                        float[] centerVec = centers[c];
                        for (int d = 0; d < dim; d++) {
                            centerVec[d] *= inv;
                        }
                    }
                    // count == 1 already equals the mean; count == 0 is handled by fillEmpty.
                }
            }

            // Fill empty clusters from Hamming-nearest non-empty cluster (document mode only).
            if (fillEmpty) {
                for (int c = 0; c < numPartitions; c++) {
                    if (counts[c] != 0) {
                        continue;
                    }
                    int nearest = -1;
                    int minDist = Integer.MAX_VALUE;
                    for (int other = 0; other < numPartitions; other++) {
                        if (counts[other] > 0) {
                            int dist = hammingDistance(c, other);
                            if (dist < minDist) {
                                minDist = dist;
                                nearest = other;
                            }
                        }
                    }
                    if (nearest >= 0) {
                        int vecIdx = clusterVecIndices[nearest][0];
                        System.arraycopy(vectors[vecIdx], 0, centers[c], 0, dim);
                    }
                }
            }

            // Random projection — each row is a contiguous float[dim] so VectorUtil.dotProduct
            // can SIMD-accelerate the FMA.
            float[][] projection = dimReductionProjections[r];
            for (int c = 0; c < numPartitions; c++) {
                float[] centerVec = centers[c];
                for (int j = 0; j < dimProj; j++) {
                    output[outOffset++] = scale * VectorUtil.dotProduct(centerVec, projection[j]);
                }
            }
        }
        return output;
    }
}

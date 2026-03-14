/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.processor.muvera;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * MUVERA (Multi-Vector Retrieval via Fixed Dimensional Encodings)
 *
 * Converts variable-length multi-vector embeddings (e.g. ColBERT/ColPali token embeddings)
 * into fixed-size single vectors that approximate MaxSim scoring via dot product.
 *
 * The output FDE dimension = rReps * 2^kSim * dimProj.
 *
 * Document processing: normalizes cluster centers by count, fills empty clusters via Hamming nearest neighbor.
 * Query processing: raw sum (no normalization), no empty cluster filling.
 *
 * The random seed must be identical between index time and query time to produce compatible encodings.
 */
public class MuveraEncoder {
    private final int dim;
    private final int kSim;
    private final int dimProj;
    private final int rReps;
    private final int numPartitions;
    private final double[][] simhashVectors;
    private final double[][][] dimReductionProjections;

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

        this.dim = dim;
        this.kSim = kSim;
        this.dimProj = dimProj;
        this.rReps = rReps;
        this.numPartitions = 1 << kSim;

        Random rng = new Random(seed);

        // SimHash hyperplanes: rReps sets of kSim hyperplanes, each dim-dimensional
        simhashVectors = new double[rReps][kSim * dim];
        for (int r = 0; r < rReps; r++) {
            for (int i = 0; i < kSim * dim; i++) {
                simhashVectors[r][i] = rng.nextGaussian();
            }
        }

        // Random projection matrices with entries in {-1, +1}
        dimReductionProjections = new double[rReps][dim][dimProj];
        for (int r = 0; r < rReps; r++) {
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < dimProj; j++) {
                    dimReductionProjections[r][i][j] = rng.nextBoolean() ? 1.0 : -1.0;
                }
            }
        }
    }

    /**
     * Returns the output FDE dimension: rReps * 2^kSim * dimProj.
     */
    public int getEmbeddingSize() {
        return rReps * numPartitions * dimProj;
    }

    private int getClusterId(double[] vector, int repIndex) {
        int clusterId = 0;
        for (int k = 0; k < kSim; k++) {
            double dot = 0;
            int offset = k * dim;
            for (int d = 0; d < this.dim; d++) {
                dot += vector[d] * simhashVectors[repIndex][offset + d];
            }
            if (dot > 0) {
                clusterId |= (1 << k);
            }
        }
        return clusterId;
    }

    private int hammingDistance(int a, int b) {
        return Integer.bitCount(a ^ b);
    }

    /**
     * Core FDE encoding.
     *
     * @param vectors    multi-vector input, shape [numVectors][dim]
     * @param fillEmpty  if true, fill empty clusters from Hamming-nearest non-empty cluster (document mode)
     * @param normalize  if true, normalize cluster centers by count (document mode)
     * @return FDE vector of length getEmbeddingSize()
     */
    public float[] process(double[][] vectors, boolean fillEmpty, boolean normalize) {
        int nVectors = vectors.length;
        float[] output = new float[getEmbeddingSize()];
        int outOffset = 0;
        double scale = 1.0 / Math.sqrt(dimProj);

        for (int r = 0; r < rReps; r++) {
            double[][] centers = new double[numPartitions][dim];
            int[] counts = new int[numPartitions];
            List<List<Integer>> clusterVecIndices = new ArrayList<>(numPartitions);
            for (int i = 0; i < numPartitions; i++) {
                clusterVecIndices.add(new ArrayList<>());
            }

            // Assign vectors to clusters and accumulate
            for (int v = 0; v < nVectors; v++) {
                int cid = getClusterId(vectors[v], r);
                counts[cid]++;
                clusterVecIndices.get(cid).add(v);
                for (int d = 0; d < dim; d++) {
                    centers[cid][d] += vectors[v][d];
                }
            }

            // Normalize by count (document mode)
            if (normalize) {
                for (int c = 0; c < numPartitions; c++) {
                    if (counts[c] > 0) {
                        for (int d = 0; d < dim; d++) {
                            centers[c][d] /= counts[c];
                        }
                    }
                }
            }

            // Fill empty clusters from Hamming-nearest non-empty cluster (document mode)
            if (fillEmpty) {
                for (int c = 0; c < numPartitions; c++) {
                    if (counts[c] == 0) {
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
                            int vecIdx = clusterVecIndices.get(nearest).get(0);
                            System.arraycopy(vectors[vecIdx], 0, centers[c], 0, dim);
                        }
                    }
                }
            }

            // Random projection: centers[c] (dim) x projection (dim x dimProj) -> projected (dimProj)
            for (int c = 0; c < numPartitions; c++) {
                for (int j = 0; j < dimProj; j++) {
                    double val = 0;
                    for (int d = 0; d < dim; d++) {
                        val += centers[c][d] * dimReductionProjections[r][d][j];
                    }
                    output[outOffset++] = (float) (scale * val);
                }
            }
        }
        return output;
    }

    /**
     * Encode document multi-vectors into a single FDE vector.
     * Normalizes cluster centers and fills empty clusters.
     */
    public float[] processDocument(double[][] vectors) {
        return process(vectors, true, true);
    }

    /**
     * Encode query multi-vectors into a single FDE vector.
     * No normalization, no empty cluster filling.
     */
    public float[] processQuery(double[][] vectors) {
        return process(vectors, false, false);
    }
}

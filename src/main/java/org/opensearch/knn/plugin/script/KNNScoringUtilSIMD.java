/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

final class KNNScoringUtilSIMD {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> INTEGER_VECTOR_SPECIES = IntVector.SPECIES_PREFERRED;

    // SHIFT_VECTOR is used to extract individual bits from packed bytes.
    // It holds the shift offsets required to move each bit to the least significant bit (LSB) position.
    // Shift vector [7, 6, 5, 4, 3, 2, 1, 0, 15, 14...]
    private static final IntVector SHIFT_VECTOR;
    static {
        // Calculate shift offset combining byte's starting bit and its relative reverse bit position
        int lanes = INTEGER_VECTOR_SPECIES.length();
        int[] shiftOffsets = new int[lanes];
        for (int i = 0; i < lanes; i++) {
            int byteBaseOffset = (i / 8) * 8;
            int bitOffset = 7 - (i % 8);
            shiftOffsets[i] = byteBaseOffset + bitOffset;
        }
        SHIFT_VECTOR = IntVector.fromArray(INTEGER_VECTOR_SPECIES, shiftOffsets, 0);
    }

    /**
     * Unpacks bits from inputVector starting at dimension i into a FloatVector.
     * Packs (step/8) consecutive bytes into a single int, broadcasts across all lanes,
     * then isolates each bit via shift and mask before converting to float (0.0f or 1.0f).
     *
     * Example
     *      lanes=8:  packedBits = [byte0]
     *      lanes=16: packedBits = [byte1 | byte0]
     *      lanes=32: packedBits = [byte3 | byte2 | byte1 | byte0]
     *
     * @param inputVector The compressed binary vector where each bit represents a dimension
     * @param i           The current dimension index (must be a multiple of step)
     * @param step        The number of dimensions processed per SIMD iteration (equals SPECIES.length())
     * @return FloatVector where each lane contains 0.0f or 1.0f corresponding to the extracted bit
     */
    private static FloatVector unpackBitsToFloatVector(byte[] inputVector, int i, int step){
        int byteIndex  = i / 8;
        int packedBits = 0;
        for (int j = 0; j < (step / 8); j++) {
            // Extract an unsigned byte and merge it into packedBits
            // at the correct 8-bit offset (0, 8, 16...).
            int byteValue = inputVector[byteIndex + j] & 0xFF;
            int bitOffset = j * 8;
            packedBits |= byteValue << bitOffset;
        }

        // 1. Broadcast packedBits to all lanes
        // 2. LSHR by SHIFT_VECTOR to move each target bit to position 0
        // 3. AND with 1 to isolate the bit (0 or 1)
        // 4. Convert int 0/1 to float 0.0f/1.0f
        return (FloatVector) IntVector.broadcast(INTEGER_VECTOR_SPECIES, packedBits)
                .lanewise(VectorOperators.LSHR, SHIFT_VECTOR)
                .lanewise(VectorOperators.AND, 1)
                .convert(VectorOperators.I2F, 0);
    }

    /**
     * Calculates the L2 squared distance between a float query vector and a binary document vector using ADC (Asymmetric Distance Computation).
     * This method implements a specialized version of L2 distance calculation where one vector is in binary format (compressed)
     * and the other is in float format (uncompressed).
     * Uses SIMD (FloatVector.SPECIES_PREFERRED) for vectorized processing with reduceLanes().
     *
     * @param queryVector The uncompressed query vector in float format
     * @param inputVector The compressed document vector in binary format, where each bit represents a dimension
     * @return The L2 squared distance between the two vectors. Lower values indicate closer vectors.
     * @throws IllegalArgumentException if queryVector length is not compatible with inputVector length (queryVector.length != inputVector.length * 8)
     */
    public static float l2SquaredADC(float[] queryVector, byte[] inputVector) {
        final int length = queryVector.length;
        final int step = SPECIES.length();
        final int loopBound = SPECIES.loopBound(length);

        FloatVector distanceAccumulator = FloatVector.zero(SPECIES);
        int i = 0;

        for (; i < loopBound; i += step) {
            FloatVector queryFloatVector = FloatVector.fromArray(SPECIES, queryVector, i);
            FloatVector floatedBits = unpackBitsToFloatVector(inputVector, i, step);

            // Compute squared difference and accumulate: acc += (bit - query)^2
            FloatVector diff = floatedBits.sub(queryFloatVector);
            distanceAccumulator = diff.fma(diff, distanceAccumulator);
        }

        // Horizontal sum of all SIMD lanes
        float score = distanceAccumulator.reduceLanes(VectorOperators.ADD);

        // Tail loop for any remaining dimensions that didn't fit into the SIMD lanes
        for (; i < queryVector.length; i++) {
            int byteIndex = i / 8;
            int bitOffset = 7 - (i % 8);
            float bitValue = (inputVector[byteIndex] >> bitOffset) & 1;

            // Calculate squared difference
            float diff = bitValue - queryVector[i];
            score += diff * diff;
        }
        return score;
    }

    /**
     * Calculates the inner product similarity between a float query vector and a binary document vector using ADC
     * (Asymmetric Distance Computation). This method is useful for similarity searches where one vector is compressed
     * in binary format and the other remains in float format.
     * Uses SIMD (FloatVector.SPECIES_PREFERRED) for vectorized processing with reduceLanes().
     *
     * The inner product is calculated by summing the products of corresponding elements, where the binary vector's
     * elements are interpreted as 0 or 1.
     *
     * @param queryVector The uncompressed query vector in float format
     * @param inputVector The compressed document vector in binary format, where each bit represents a dimension
     * @return The inner product similarity score between the two vectors. Higher values indicate more similar vectors.
     * @throws IllegalArgumentException if queryVector length is not compatible with inputVector length (queryVector.length != inputVector.length * 8)
     */
    public static float innerProductADC(float[] queryVector, byte[] inputVector) {
        final int length = queryVector.length;
        final int step = SPECIES.length();
        final int loopBound = SPECIES.loopBound(length);
        FloatVector distanceAccumulator = FloatVector.zero(SPECIES);
        int i = 0;

        for (; i < loopBound; i += step) {
            FloatVector queryFloatVector = FloatVector.fromArray(SPECIES, queryVector, i);
            FloatVector floatedBits = unpackBitsToFloatVector(inputVector, i, step);

            // Compute product and accumulate: acc += (bit * query)
            distanceAccumulator = floatedBits.fma(queryFloatVector, distanceAccumulator);
        }

        // Horizontal sum of all SIMD lanes
        float score = distanceAccumulator.reduceLanes(VectorOperators.ADD);

        // Tail loop for any remaining dimensions that didn't fit into the SIMD lanes
        for (; i < queryVector.length; i++) {
            // Extract the bit for this dimension
            int byteIndex = i / 8;
            int bitOffset = 7 - (i % 8);
            int bitValue = (inputVector[byteIndex] >> bitOffset) & 1;

            // Calculate product and accumulate
            score += bitValue * queryVector[i];
        }
        return score;
    }

}

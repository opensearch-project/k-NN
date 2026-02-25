/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.generate;

import org.apache.lucene.util.VectorUtil;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.VectorDataType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.LongStream;

public class SearchTestHelper {
    public static class Vectors {
        public final VectorDataType dataType;
        public final List<float[]> floatVectors;
        public final List<byte[]> byteVectors;

        public Vectors(final List<float[]> floatVectors) {
            this.dataType = VectorDataType.FLOAT;
            this.floatVectors = floatVectors;
            this.byteVectors = null;
        }

        public Vectors(final VectorDataType dataType, final List<byte[]> byteVectors) {
            this.dataType = dataType;
            this.floatVectors = null;
            this.byteVectors = byteVectors;
        }

        public static Vectors create(final VectorDataType dataType) {
            if (dataType == VectorDataType.FLOAT) {
                return new Vectors(new ArrayList<>());
            } else if (dataType == VectorDataType.BYTE) {
                return new Vectors(dataType, new ArrayList<>());
            } else {
                throw new AssertionError();
            }
        }
    }

    /**
     * Extract parent ids from the given child document ids.
     * Parent id is marked as null in the doc id list, from there it collects its corresponding indices which is parent document id.
     * Ex: With [0,1,2,3,4,  6,7,8,9,10   22,23,24,25,26], parent ids=[5, 14, 27] with 5 child docs.
     *
     * @param childDocIds Child document ids
     * @return Parent document ids
     */
    public static int[] extractParentIds(final List<Integer> childDocIds) {
        final List<Integer> parentIds = new ArrayList<>();
        for (int i = 0, j = childDocIds.get(0); i < childDocIds.size();) {
            if (childDocIds.get(i) != j) {
                parentIds.add(j);
                j = childDocIds.get(i);
            } else {
                ++i;
                ++j;
            }
        }

        // We always add child docs at the end.
        // So we can safely add parent id here.
        // Ex: With [, ..., 100, 101, 102, 103, 104], parent id would be 105 (e.g. having 5 child docs)
        parentIds.add(childDocIds.get(childDocIds.size() - 1) + 1);
        return parentIds.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Take X% of the given integer sequence. After the selection, int values will be sorted then be returned.
     *
     * @param sequence An integer sequence.
     * @param percentage X% portion to select from the integer sequence.
     * @return Selected portion of integer sequence.
     */
    public static List<Integer> takePortions(final List<Integer> sequence, final double percentage) {
        List<Integer> newSequence = new ArrayList<>(sequence);
        Collections.shuffle(newSequence);
        newSequence = newSequence.subList(0, (int) (newSequence.size() * percentage));
        Collections.sort(newSequence);
        return newSequence;
    }

    /**
     * Util method to convert List<byte[]> to List<float[]>
     *
     * @param byteVectors Byte vectors
     * @return Float vectors in List<float[]>
     */
    public static List<float[]> convertToFloatList(final List<byte[]> byteVectors) {
        List<float[]> vectors = new ArrayList<>();
        for (byte[] bytes : byteVectors) {
            if (bytes == null) {
                vectors.add(null);
            } else {
                vectors.add(convertToFloatArray(bytes));
            }
        }
        return vectors;
    }

    /**
     * Util method to convert byte[] to int[]
     *
     * @param byteVector Byte vectors
     * @return Integer vectors in int[]
     */
    public static int[] convertToIntArray(final byte[] byteVector) {
        final int dimensions = byteVector.length;
        final int[] ints = new int[dimensions];
        for (int i = 0; i < dimensions; i++) {
            ints[i] = byteVector[i];
        }
        return ints;
    }

    /**
     * Util method to convert byte[] to float[]
     *
     * @param byteVector Byte vectors
     * @return Float vectors in float[]
     */
    public static float[] convertToFloatArray(final byte[] byteVector) {
        final int dimensions = byteVector.length;
        final float[] floats = new float[dimensions];
        for (int i = 0; i < dimensions; i++) {
            floats[i] = byteVector[i];
        }
        return floats;
    }

    /**
     * Util method to convert float[] to byte[]
     *
     * @param vector Float vectors
     * @return Byte vectors in byte[]
     */
    public static byte[] convertToByteArray(float[] vector) {
        byte[] bytes = new byte[vector.length];
        for (int i = 0; i < vector.length; i++) {
            bytes[i] = (byte) vector[i]; // cast float to byte
        }
        return bytes;
    }

    /**
     * Get document ids list that is expected to be obtained with an exact search.
     *
     * @param givenDocumentIds Entire document ids.
     * @param vectors Entire vectors
     * @param query Query vector. float[] or byte[]
     * @param filteredIds Filtered document ids.
     * @param similarityFunction Similarity function.
     * @param topK Top-k value of result.
     * @return Return document ids that is expected to be obtained with an exact search.
     */
    public static Set<Integer> getKnnAnswerSetForVectors(
        final List<Integer> givenDocumentIds,
        final Vectors vectors,
        final Object query,
        final long[] filteredIds,
        final KNNVectorSimilarityFunction similarityFunction,
        final int topK
    ) {
        List<Integer> documentIds = givenDocumentIds;

        final Set<Integer> filteredIdSet;
        if (filteredIds != null) {
            filteredIdSet = LongStream.of(filteredIds).mapToInt(l -> (int) l).boxed().collect(Collectors.toSet());
        } else {
            filteredIdSet = null;
        }

        // Create a new one
        documentIds = new ArrayList<>(documentIds);

        // Sort indices by similarity
        if (similarityFunction != KNNVectorSimilarityFunction.HAMMING) {
            List<float[]> floatVectors = vectors.floatVectors;
            if (vectors.dataType != VectorDataType.FLOAT) {
                assert vectors.byteVectors != null;
                floatVectors = convertToFloatList(vectors.byteVectors);
            }

            final List<float[]> finalFloats = floatVectors;
            documentIds.sort(
                (docId1, docId2) -> -Float.compare(
                    similarityFunction.compare(finalFloats.get(docId1), (float[]) query),
                    similarityFunction.compare(finalFloats.get(docId2), (float[]) query)
                )
            );
        } else {
            assert (query instanceof byte[]);
            documentIds.sort(
                (docId1, docId2) -> -Float.compare(
                    similarityFunction.compare(vectors.byteVectors.get(docId1), (byte[]) query),
                    similarityFunction.compare(vectors.byteVectors.get(docId2), (byte[]) query)
                )
            );
        }

        // Apply filtering and collect top K
        final Set<Integer> answerSet = new HashSet<>();
        int i = 0;
        while (answerSet.size() < topK && i < documentIds.size()) {
            if (filteredIdSet == null || filteredIdSet.contains(documentIds.get(i))) {
                answerSet.add(documentIds.get(i));
            }
            ++i;
        }

        return answerSet;
    }

    /**
     * Generate random byte vectors. Resulting vectors will have the same size as the given document ids.
     *
     * @param docIds Document ids.
     * @param dimensions Vector dimension.
     * @param minValue Min value of vector element.
     * @param maxValue Max value of vector element.
     * @return Byte vectors.
     */
    public static List<byte[]> generateRandomByteVectors(
        final List<Integer> docIds,
        final int dimensions,
        final float minValue,
        final float maxValue
    ) {

        final List<byte[]> vectors = new ArrayList<>();

        for (int i = 0, j = 0; j < docIds.size(); j++) {
            // Add null vectors.
            // e.g. previous doc=15, current doc=18.
            // then put two nulls for doc=16, 17. This indicates that doc=16 and 17 don't have vector field.
            while (i < docIds.get(j)) {
                vectors.add(null);
                ++i;
            }

            vectors.add(generateOneSingleByteVector(dimensions, minValue, maxValue));
            ++i;
        }

        return vectors;
    }

    public static float[] generateOneSingleFloatVector(final int dimension, final float minValue, float maxValue, final boolean normalize) {
        final float[] vector = new float[dimension];
        for (int k = 0; k < dimension; k++) {
            vector[k] = minValue + ThreadLocalRandom.current().nextFloat() * (maxValue - minValue);
        }
        if (normalize) {
            return VectorUtil.l2normalize(vector);
        }
        return vector;
    }

    /**
     * Generate random float vectors. Resulting vectors will have the same size as the given document ids.
     *
     * @param docIds Document ids.
     * @param dimensions Vector dimension.
     * @param minValue Min value of vector element.
     * @param maxValue Max value of vector element.
     * @return Float vectors.
     */
    public static List<float[]> generateRandomFloatVectors(
        final List<Integer> docIds,
        final int dimensions,
        final float minValue,
        final float maxValue,
        final boolean normalize
    ) {

        final List<float[]> vectors = new ArrayList<>();
        for (int i = 0, j = 0; j < docIds.size(); j++) {
            // Add null vectors.
            // e.g. previous doc=15, current doc=18.
            // then put two nulls for doc=16, 17. This indicates that doc=16 and 17 don't have vector field.
            while (i < docIds.get(j)) {
                vectors.add(null);
                ++i;
            }

            vectors.add(generateOneSingleFloatVector(dimensions, minValue, maxValue, normalize));
            ++i;
        }

        return vectors;
    }

    public static byte[] generateOneSingleByteVector(final int dimension, final float minValue, float maxValue) {
        final byte[] vector = new byte[dimension];
        for (int k = 0; k < dimension; k++) {
            vector[k] = (byte) (minValue + ThreadLocalRandom.current().nextInt((int) (maxValue - minValue + 1)));
        }
        return vector;
    }

    public static byte[] generateOneSingleBinaryVector(final int dimension) {
        assert dimension % 8 == 0;

        // Making a binary vector
        // Ex: Dimension = 16, then it will require 16 bits, 2 bytes.
        // [48, 77] -> [0b00110000, 0b01001101] -> a binary vector [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1]

        final int numBytes = dimension / 8;
        final byte[] vector = new byte[numBytes];
        for (int k = 0; k < numBytes; k++) {
            vector[k] = (byte) (ThreadLocalRandom.current().nextInt(Byte.MIN_VALUE, Byte.MAX_VALUE));
        }
        return vector;
    }
}

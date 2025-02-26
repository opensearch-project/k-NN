/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.vectorvalues;

import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.index.codec.util.KNNVectorSerializer;
import org.opensearch.knn.index.codec.util.KNNVectorSerializerFactory;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static com.carrotsearch.randomizedtesting.RandomizedTest.randomByte;
import static com.carrotsearch.randomizedtesting.RandomizedTest.randomFloat;

public class TestVectorValues {

    public static class RandomVectorBinaryDocValues extends VectorDocValues {

        public RandomVectorBinaryDocValues(int count, int dimension) {
            super(count, dimension);
        }

        @Override
        public BytesRef binaryValue() throws IOException {
            return new BytesRef(knnVectorSerializer.floatToByteArray(getRandomVector(dimension)));
        }
    }

    public static class ConstantVectorBinaryDocValues extends VectorDocValues {

        private final BytesRef value;

        public ConstantVectorBinaryDocValues(int count, int dimension, float value) {
            super(count, dimension);
            float[] array = new float[dimension];
            Arrays.fill(array, value);
            this.value = new BytesRef(knnVectorSerializer.floatToByteArray(array));
        }

        @Override
        public BytesRef binaryValue() throws IOException {
            return value;
        }
    }

    public static class PredefinedFloatVectorBinaryDocValues extends VectorDocValues {
        private final List<float[]> vectors;

        public PredefinedFloatVectorBinaryDocValues(final List<float[]> vectors) {
            super(vectors.size(), vectors.get(0).length);
            this.vectors = vectors;
        }

        @Override
        public BytesRef binaryValue() throws IOException {
            return new BytesRef(knnVectorSerializer.floatToByteArray(vectors.get(docID())));
        }
    }

    public static class PredefinedByteVectorBinaryDocValues extends VectorDocValues {
        private final List<byte[]> vectors;

        public PredefinedByteVectorBinaryDocValues(final List<byte[]> vectors) {
            super(vectors.size(), vectors.get(0).length);
            this.vectors = vectors;
        }

        @Override
        public BytesRef binaryValue() throws IOException {
            return new BytesRef(vectors.get(docID()));
        }
    }

    public static class RandomVectorDocValuesProducer extends DocValuesProducer {

        final RandomVectorBinaryDocValues randomBinaryDocValues;

        public RandomVectorDocValuesProducer(int count, int dimension) {
            this.randomBinaryDocValues = new RandomVectorBinaryDocValues(count, dimension);
        }

        @Override
        public NumericDocValues getNumeric(FieldInfo field) {
            return null;
        }

        @Override
        public BinaryDocValues getBinary(FieldInfo field) throws IOException {
            return randomBinaryDocValues;
        }

        @Override
        public SortedDocValues getSorted(FieldInfo field) {
            return null;
        }

        @Override
        public SortedNumericDocValues getSortedNumeric(FieldInfo field) {
            return null;
        }

        @Override
        public SortedSetDocValues getSortedSet(FieldInfo field) {
            return null;
        }

        @Override
        public void checkIntegrity() {

        }

        @Override
        public void close() throws IOException {

        }
    }

    static abstract class VectorDocValues extends BinaryDocValues {

        final int count;
        final int dimension;
        int current;
        KNNVectorSerializer knnVectorSerializer;

        public VectorDocValues(int count, int dimension) {
            this.count = count;
            this.dimension = dimension;
            this.current = -1;
            this.knnVectorSerializer = KNNVectorSerializerFactory.getDefaultSerializer();
        }

        @Override
        public boolean advanceExact(int target) throws IOException {
            return false;
        }

        @Override
        public int docID() {
            if (this.current > this.count) {
                return BinaryDocValues.NO_MORE_DOCS;
            }
            return this.current;
        }

        @Override
        public int nextDoc() throws IOException {
            return advance(current + 1);
        }

        @Override
        public int advance(int target) throws IOException {
            current = target;
            if (current >= count) {
                current = NO_MORE_DOCS;
            }
            return current;
        }

        @Override
        public long cost() {
            return count;
        }
    }

    public static class PreDefinedFloatVectorValues extends FloatVectorValues {
        final int count;
        final int dimension;
        final List<float[]> vectors;
        int current;
        float[] vector;

        public PreDefinedFloatVectorValues(final List<float[]> vectors) {
            super();
            this.count = vectors.size();
            if (!vectors.isEmpty()) {
                this.dimension = vectors.get(0).length;
            } else {
                this.dimension = 0;
            }
            this.vectors = vectors;
            this.current = -1;
            vector = new float[dimension];
        }

        @Override
        public int dimension() {
            return dimension;
        }

        @Override
        public int size() {
            return count;
        }

        @Override
        public float[] vectorValue() throws IOException {
            // since in FloatVectorValues the reference to returned vector doesn't change. This code ensure that we
            // are replicating the behavior so that if someone uses this RandomFloatVectorValues they get an
            // experience similar to what we get in prod.
            System.arraycopy(vectors.get(docID()), 0, vector, 0, dimension);
            return vector;
        }

        @Override
        public VectorScorer scorer(float[] query) throws IOException {
            throw new UnsupportedOperationException("scorer not supported with PreDefinedFloatVectorValues");
        }

        @Override
        public int docID() {
            if (this.current > this.count) {
                return FloatVectorValues.NO_MORE_DOCS;
            }
            return this.current;
        }

        @Override
        public int nextDoc() throws IOException {
            return advance(current + 1);
        }

        @Override
        public int advance(int target) throws IOException {
            current = target;
            if (current >= count) {
                current = NO_MORE_DOCS;
            }
            return current;
        }
    }

    public static class PreDefinedByteVectorValues extends ByteVectorValues {
        private final int count;
        private final int dimension;
        private final List<byte[]> vectors;
        private int current;
        private final byte[] vector;

        public PreDefinedByteVectorValues(final List<byte[]> vectors) {
            super();
            this.count = vectors.size();
            this.dimension = vectors.get(0).length;
            this.vectors = vectors;
            this.current = -1;
            vector = new byte[dimension];
        }

        @Override
        public int dimension() {
            return dimension;
        }

        @Override
        public int size() {
            return count;
        }

        @Override
        public byte[] vectorValue() throws IOException {
            // since in FloatVectorValues the reference to returned vector doesn't change. This code ensure that we
            // are replicating the behavior so that if someone uses this RandomFloatVectorValues they get an
            // experience similar to what we get in prod.
            System.arraycopy(vectors.get(docID()), 0, vector, 0, dimension);
            return vector;
        }

        @Override
        public VectorScorer scorer(byte[] query) throws IOException {
            throw new UnsupportedOperationException("scorer not supported with PreDefinedFloatVectorValues");
        }

        @Override
        public int docID() {
            if (this.current > this.count) {
                return FloatVectorValues.NO_MORE_DOCS;
            }
            return this.current;
        }

        @Override
        public int nextDoc() throws IOException {
            return advance(current + 1);
        }

        @Override
        public int advance(int target) throws IOException {
            current = target;
            if (current >= count) {
                current = NO_MORE_DOCS;
            }
            return current;
        }
    }

    public static class PreDefinedBinaryVectorValues extends PreDefinedByteVectorValues {

        public PreDefinedBinaryVectorValues(List<byte[]> vectors) {
            super(vectors);
        }

        @Override
        public int dimension() {
            return super.dimension() * Byte.SIZE;
        }
    }

    public static class NotBinaryDocValues extends NumericDocValues {

        @Override
        public long longValue() throws IOException {
            return 0;
        }

        @Override
        public boolean advanceExact(int target) throws IOException {
            return false;
        }

        @Override
        public int docID() {
            return 0;
        }

        @Override
        public int nextDoc() throws IOException {
            return 0;
        }

        @Override
        public int advance(int target) throws IOException {
            return 0;
        }

        @Override
        public long cost() {
            return 0;
        }
    }

    public static float[][] getRandomVectors(int count, int dimension) {
        float[][] data = new float[count][dimension];
        for (int i = 0; i < count; i++) {
            data[i] = getRandomVector(dimension);
        }
        return data;
    }

    public static float[] getRandomVector(int dimension) {
        float[] data = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            data[i] = randomFloat();
        }
        return data;
    }

    public static byte[] getRandomByteVector(int dimension) {
        byte[] data = new byte[dimension];
        for (int i = 0; i < dimension; i++) {
            data[i] = randomByte();
        }
        return data;
    }
}

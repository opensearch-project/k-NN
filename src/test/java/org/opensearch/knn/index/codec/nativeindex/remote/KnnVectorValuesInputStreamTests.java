/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;
import static org.opensearch.knn.index.vectorvalues.TestVectorValues.getRandomByteVector;
import static org.opensearch.knn.index.vectorvalues.TestVectorValues.getRandomVector;

public class KnnVectorValuesInputStreamTests extends KNNTestCase {

    /**
     * Tests that reading doc IDs out of a DocIdInputStream yields the same results as reading the doc ids
     */
    public void testDocIdInputStream() throws IOException {
        int NUM_DOCS = randomIntBetween(1, 1000);

        List<float[]> vectorValues = getRandomFloatVectors(NUM_DOCS, 1);
        final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            vectorValues
        );
        final KNNVectorValues<float[]> knnVectorValuesForStream = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT,
            randomVectorValues
        );
        final KNNVectorValues<float[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, randomVectorValues);

        InputStream docIdInputStream = new DocIdInputStream(knnVectorValuesForStream);

        // 1. Read all input stream bytes
        byte[] docIdStreamBytes = docIdInputStream.readAllBytes();

        // 2. Read all of knnVectorValues into a byte buffer:
        ByteBuffer buffer = ByteBuffer.allocate(NUM_DOCS * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        initializeVectorValues(knnVectorValues);
        int docId = knnVectorValues.docId();
        while (docId != -1 && docId != DocIdSetIterator.NO_MORE_DOCS) {
            buffer.putInt(docId);
            docId = knnVectorValues.nextDoc();
        }

        // Check the 2 arrays have the same content
        assertArrayEquals(docIdStreamBytes, buffer.array());
    }

    /**
     * Tests that reading float vectors out of a VectorValuesInputStream yields the same results as reading the doc vectors
     */
    public void testFloatVectorValuesInputStream() throws IOException {
        int NUM_DOCS = randomIntBetween(1, 1000);
        int NUM_DIMENSION = randomIntBetween(1, 1000);

        List<float[]> vectorValues = getRandomFloatVectors(NUM_DOCS, NUM_DIMENSION);
        final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            vectorValues
        );
        final KNNVectorValues<float[]> knnVectorValuesForStream = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT,
            randomVectorValues
        );
        final KNNVectorValues<float[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, randomVectorValues);

        InputStream vectorValuesInputStream = new VectorValuesInputStream(knnVectorValuesForStream, VectorDataType.FLOAT);

        // 1. Read all input stream bytes
        byte[] vectorStreamBytes = vectorValuesInputStream.readAllBytes();
        FloatBuffer vectorStreamFloats = ByteBuffer.wrap(vectorStreamBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();

        // 2. Read all of knnVectorValues into a byte buffer:
        initializeVectorValues(knnVectorValues);
        FloatBuffer expectedBuffer = ByteBuffer.allocate(NUM_DOCS * knnVectorValues.bytesPerVector())
            .order(ByteOrder.LITTLE_ENDIAN)
            .asFloatBuffer();
        int docId = knnVectorValues.docId();
        while (docId != -1 && docId != DocIdSetIterator.NO_MORE_DOCS) {
            expectedBuffer.put(knnVectorValues.getVector());
            docId = knnVectorValues.nextDoc();
        }
        expectedBuffer.position(0);

        // Check the 2 arrays have the same content
        assertEquals(expectedBuffer, vectorStreamFloats);
    }

    /*
     * Tests that reading half-float vectors out of a VectorValuesInputStream yields the same results as reading the doc vectors
     */
    public void testHalfFloatVectorValuesInputStream() throws IOException {
        int NUM_DOCS = randomIntBetween(1, 1000);
        int NUM_DIMENSION = randomIntBetween(1, 1000);

        List<float[]> vectorValues = getRandomFloatVectors(NUM_DOCS, NUM_DIMENSION);
        final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            vectorValues
        );
        final KNNVectorValues<float[]> knnVectorValuesForStream = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.HALF_FLOAT,
            randomVectorValues
        );
        final KNNVectorValues<float[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.HALF_FLOAT,
            randomVectorValues
        );

        InputStream vectorValuesInputStream = new VectorValuesInputStream(knnVectorValuesForStream, VectorDataType.HALF_FLOAT);

        // 1. Read all input stream bytes
        byte[] vectorStreamBytes = vectorValuesInputStream.readAllBytes();
        FloatBuffer vectorStreamFloats = ByteBuffer.wrap(vectorStreamBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();

        // 2. Read all of knnVectorValues into a byte buffer:
        initializeVectorValues(knnVectorValues);
        FloatBuffer expectedBuffer = ByteBuffer.allocate(NUM_DOCS * knnVectorValues.bytesPerVector())
            .order(ByteOrder.LITTLE_ENDIAN)
            .asFloatBuffer();
        int docId = knnVectorValues.docId();
        while (docId != -1 && docId != DocIdSetIterator.NO_MORE_DOCS) {
            expectedBuffer.put(knnVectorValues.getVector());
            docId = knnVectorValues.nextDoc();
        }
        expectedBuffer.position(0);

        // Check the 2 arrays have the same content
        assertEquals(expectedBuffer, vectorStreamFloats);
    }

    /**
     * Tests that invoking {@link VectorValuesInputStream#read()} N times yields the same results as {@link VectorValuesInputStream#read(byte[], 0, N)}
     */
    public void testHalfFloatVectorValuesInputStreamReadByte() throws IOException {
        final int NUM_DIMENSION = randomIntBetween(1, 1000);
        List<float[]> vectorValues = getRandomFloatVectors(1, NUM_DIMENSION);

        final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            vectorValues
        );

        // Read stream byte by byte
        final KNNVectorValues<float[]> knnVectorValuesForReadByte = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.HALF_FLOAT,
            randomVectorValues
        );
        initializeVectorValues(knnVectorValuesForReadByte);
        int vectorBlobLength = knnVectorValuesForReadByte.bytesPerVector();
        InputStream vectorStreamForReadByte = new VectorValuesInputStream(knnVectorValuesForReadByte, VectorDataType.HALF_FLOAT);
        ByteBuffer bufferReadByByte = ByteBuffer.allocate(vectorBlobLength).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < vectorBlobLength; i++) {
            bufferReadByByte.put((byte) vectorStreamForReadByte.read());
        }
        bufferReadByByte.position(0);

        // Read stream with entire length
        final KNNVectorValues<float[]> knnVectorValuesForRead = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.HALF_FLOAT,
            randomVectorValues
        );
        InputStream vectorStreamForRead = new VectorValuesInputStream(knnVectorValuesForRead, VectorDataType.HALF_FLOAT);
        ByteBuffer bufferRead = ByteBuffer.allocate(vectorBlobLength).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(vectorBlobLength, vectorStreamForRead.read(bufferRead.array(), 0, vectorBlobLength));

        assertArrayEquals(bufferRead.array(), bufferReadByByte.array());
    }

    public void testByteVectorValuesInputStream() throws IOException {
        int NUM_DOCS = randomIntBetween(1, 1000);
        int NUM_DIMENSION = randomIntBetween(1, 1000);

        List<byte[]> vectorValues = getRandomByteVectors(NUM_DOCS, NUM_DIMENSION);
        final TestVectorValues.PreDefinedByteVectorValues randomVectorValues = new TestVectorValues.PreDefinedByteVectorValues(
            vectorValues
        );
        final KNNVectorValues<byte[]> knnVectorValuesForStream = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.BYTE,
            randomVectorValues
        );
        final KNNVectorValues<byte[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.BYTE, randomVectorValues);

        InputStream vectorValuesInputStream = new VectorValuesInputStream(knnVectorValuesForStream, VectorDataType.BYTE);

        // 1. Read all input stream bytes
        byte[] vectorStreamBytes = vectorValuesInputStream.readAllBytes();

        // 2. Read all of knnVectorValues into a byte buffer:
        initializeVectorValues(knnVectorValues);
        ByteBuffer expectedBuffer = ByteBuffer.allocate(NUM_DOCS * knnVectorValues.bytesPerVector()).order(ByteOrder.LITTLE_ENDIAN);
        int docId = knnVectorValues.docId();
        while (docId != -1 && docId != DocIdSetIterator.NO_MORE_DOCS) {
            expectedBuffer.put(knnVectorValues.getVector());
            docId = knnVectorValues.nextDoc();
        }

        // Check the 2 arrays have the same content
        assertArrayEquals(expectedBuffer.array(), vectorStreamBytes);
    }

    /**
     * Tests that creating N VectorValuesInputStream over the same KNNVectorValues yields the same result as reading it all from the same VectorValuesInputStream
     */
    public void testMultiPartVectorValueInputStream() throws IOException {
        final int NUM_DOCS = randomIntBetween(100, 1000);
        final int NUM_DIMENSION = randomIntBetween(1, 1000);
        final int NUM_PARTS = randomIntBetween(1, NUM_DOCS / 10);
        final int PART_SIZE;
        final int LAST_PART_SIZE;

        List<float[]> vectorValues = getRandomFloatVectors(NUM_DOCS, NUM_DIMENSION);
        final Supplier<TestVectorValues.PreDefinedFloatVectorValues> randomVectorValuesSupplier =
            () -> new TestVectorValues.PreDefinedFloatVectorValues(vectorValues);

        final Supplier<KNNVectorValues<float[]>> knnVectorValuesSupplier = () -> KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT,
            randomVectorValuesSupplier.get()
        );

        final KNNVectorValues<float[]> knnVectorValues = knnVectorValuesSupplier.get();
        initializeVectorValues(knnVectorValues);
        int vectorBlobLength = knnVectorValues.bytesPerVector() * NUM_DOCS;
        PART_SIZE = vectorBlobLength / NUM_PARTS;
        LAST_PART_SIZE = (vectorBlobLength % PART_SIZE) != 0 ? vectorBlobLength % PART_SIZE : PART_SIZE;

        // 1. Create NUM_PARTS input streams
        final List<VectorValuesInputStream> streamList = new ArrayList<>(NUM_PARTS);
        for (int partNumber = 0; partNumber < NUM_PARTS; partNumber++) {
            System.out.println(partNumber);
            streamList.add(
                new VectorValuesInputStream(knnVectorValuesSupplier.get(), VectorDataType.FLOAT, (long) partNumber * PART_SIZE, PART_SIZE)
            );
        }

        // Last part only needs to be written if it is a different size, otherwise previous for loop would cover it
        if (LAST_PART_SIZE != PART_SIZE) {
            streamList.add(
                new VectorValuesInputStream(
                    knnVectorValuesSupplier.get(),
                    VectorDataType.FLOAT,
                    vectorBlobLength - LAST_PART_SIZE,
                    LAST_PART_SIZE
                )
            );
        }

        // 2. Read all input stream parts into the same buffer
        ByteBuffer testBuffer = ByteBuffer.allocate(vectorBlobLength).order(ByteOrder.LITTLE_ENDIAN);
        for (VectorValuesInputStream stream : streamList) {
            byte[] partBytes = stream.readAllBytes();
            testBuffer.put(partBytes);
        }

        // 3. Read all knnVectorValues into a buffer:
        VectorValuesInputStream expectedStream = new VectorValuesInputStream(knnVectorValuesSupplier.get(), VectorDataType.FLOAT);
        assertArrayEquals(expectedStream.readAllBytes(), testBuffer.array());
    }

    /**
     * Tests that invoking {@link VectorValuesInputStream#read()} N times yields the same results as {@link VectorValuesInputStream#read(byte[], 0, N)}
     */
    public void testVectorValuesInputStreamReadByte() throws IOException {
        final int NUM_DIMENSION = randomIntBetween(1, 1000);
        // We use only 1 doc here because VectorValuesInputStream.read will only read up to 1 vector maximum at a time.
        // To read all the vectors we would need to call readNBytes, however we want to specifically test the methods we have overridden
        // here
        List<float[]> vectorValues = getRandomFloatVectors(1, NUM_DIMENSION);

        final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            vectorValues
        );

        // Read stream byte by byte
        final KNNVectorValues<float[]> knnVectorValuesForReadByte = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT,
            randomVectorValues
        );
        initializeVectorValues(knnVectorValuesForReadByte);
        int vectorBlobLength = knnVectorValuesForReadByte.bytesPerVector();
        InputStream vectorStreamForReadByte = new VectorValuesInputStream(knnVectorValuesForReadByte, VectorDataType.FLOAT);
        ByteBuffer bufferReadByByte = ByteBuffer.allocate(vectorBlobLength).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < vectorBlobLength; i++) {
            bufferReadByByte.put((byte) vectorStreamForReadByte.read());
        }
        bufferReadByByte.position(0);

        // Read stream with entire length
        final KNNVectorValues<float[]> knnVectorValuesForRead = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT,
            randomVectorValues
        );
        InputStream vectorStreamForRead = new VectorValuesInputStream(knnVectorValuesForRead, VectorDataType.FLOAT);
        ByteBuffer bufferRead = ByteBuffer.allocate(vectorBlobLength).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(vectorBlobLength, vectorStreamForRead.read(bufferRead.array(), 0, vectorBlobLength));

        assertArrayEquals(bufferRead.array(), bufferReadByByte.array());
    }

    /**
     * Tests that invoking {@link DocIdInputStream#read()} N times yields the same results as {@link DocIdInputStream#read(byte[], 0, N)}
     */
    public void testDocIdInputStreamReadByte() throws IOException {
        // We use only 1 doc here because DocIdInputStream.read will only read up to 1 doc id at a time
        // To read all the vectors we would need to call readNBytes, however we want to specifically test the methods we have overridden
        // here
        List<float[]> vectorValues = getRandomFloatVectors(1, 1);

        final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            vectorValues
        );

        // Read stream byte by byte
        final KNNVectorValues<float[]> docIdsForReadByte = KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, randomVectorValues);
        initializeVectorValues(docIdsForReadByte);
        int blobLength = Integer.BYTES;
        InputStream docStreamForReadByte = new DocIdInputStream(docIdsForReadByte);
        ByteBuffer bufferReadByByte = ByteBuffer.allocate(blobLength).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < blobLength; i++) {
            bufferReadByByte.put((byte) docStreamForReadByte.read());
        }
        bufferReadByByte.position(0);

        // Read stream with entire length
        final KNNVectorValues<float[]> docIdsForRead = KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, randomVectorValues);
        InputStream docStreamForRead = new DocIdInputStream(docIdsForRead);
        ByteBuffer bufferRead = ByteBuffer.allocate(blobLength).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(blobLength, docStreamForRead.read(bufferRead.array(), 0, blobLength));

        assertArrayEquals(bufferRead.array(), bufferReadByByte.array());
    }

    private List<float[]> getRandomFloatVectors(int numDocs, int dimension) {
        ArrayList<float[]> vectorValues = new ArrayList<>();
        for (int i = 0; i < numDocs; i++) {
            vectorValues.add(getRandomVector(dimension));
        }
        return vectorValues;
    }

    private List<byte[]> getRandomByteVectors(int numDocs, int dimension) {
        ArrayList<byte[]> vectorValues = new ArrayList<>();
        for (int i = 0; i < numDocs; i++) {
            vectorValues.add(getRandomByteVector(dimension));
        }
        return vectorValues;
    }
}

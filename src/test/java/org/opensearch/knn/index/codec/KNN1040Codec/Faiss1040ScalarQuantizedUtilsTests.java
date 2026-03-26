/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.lucene104.QuantizedByteVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;

import static org.mockito.Mockito.mock;

@Log4j2
public class Faiss1040ScalarQuantizedUtilsTests extends KNNTestCase {

    /**
     * A concrete stub extending KnnVectorValues that declares the private field
     * {@code quantizedVectorValues} so that reflection-based extraction succeeds.
     */
    static class StubVectorValues extends KnnVectorValues {
        private QuantizedByteVectorValues quantizedVectorValues;

        @Override
        public int dimension() {
            return 0;
        }

        @Override
        public int size() {
            return 0;
        }

        @Override
        public KnnVectorValues copy() throws IOException {
            return this;
        }

        @Override
        public VectorEncoding getEncoding() {
            return VectorEncoding.FLOAT32;
        }
    }

    @SneakyThrows
    public void testExtractQuantizedByteVectorValues_whenFieldExists_thenReturnsValue() {
        // Arrange: create stub and set the private field via reflection
        StubVectorValues stub = new StubVectorValues();
        QuantizedByteVectorValues expected = mock(QuantizedByteVectorValues.class);

        java.lang.reflect.Field field = StubVectorValues.class.getDeclaredField("quantizedVectorValues");
        field.setAccessible(true);
        field.set(stub, expected);

        // Act
        QuantizedByteVectorValues result = Faiss1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(stub);

        // Assert: the returned reference is the exact same object
        assertSame(expected, result);
    }

    public void testExtractQuantizedByteVectorValues_whenFieldMissing_thenThrowsIOException() {
        // Arrange: a Mockito mock of KnnVectorValues lacks the quantizedVectorValues field
        KnnVectorValues mockValues = mock(KnnVectorValues.class);

        // Act & Assert
        IOException exception = expectThrows(
            IOException.class,
            () -> Faiss1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(mockValues)
        );

        assertTrue(exception.getMessage().contains("incompatible Lucene version"));
        assertTrue(exception.getCause() instanceof NoSuchFieldException);
    }
}

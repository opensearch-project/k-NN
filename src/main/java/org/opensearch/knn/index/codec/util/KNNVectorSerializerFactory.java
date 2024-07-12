/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import com.google.common.collect.ImmutableMap;

import java.io.ByteArrayInputStream;
import java.io.ObjectStreamConstants;
import java.util.Arrays;
import java.util.Map;

import static org.opensearch.knn.index.codec.util.SerializationMode.ARRAY;
import static org.opensearch.knn.index.codec.util.SerializationMode.COLLECTION_OF_FLOATS;

/**
 * Class abstracts Factory for KNNVectorSerializer implementations. Exact implementation constructed and returned based on
 * either content of the byte array or directly based on serialization type.
 */
public class KNNVectorSerializerFactory {
    private static Map<SerializationMode, KNNVectorSerializer> VECTOR_SERIALIZER_BY_TYPE = ImmutableMap.of(
        ARRAY,
        new KNNVectorAsArraySerializer(),
        COLLECTION_OF_FLOATS,
        new KNNVectorAsCollectionOfFloatsSerializer()
    );

    private static final int ARRAY_HEADER_OFFSET = 27;
    private static final int BYTES_IN_FLOAT = 4;
    private static final int BITS_IN_ONE_BYTE = 8;

    /**
     * Array represents first 6 bytes of the byte stream header as per Java serialization protocol described in details
     * <a href="https://docs.oracle.com/javase/8/docs/platform/serialization/spec/protocol.html">here</a>.
     */
    private static final byte[] SERIALIZATION_PROTOCOL_HEADER_PREFIX = new byte[] {
        highByte(ObjectStreamConstants.STREAM_MAGIC),
        lowByte(ObjectStreamConstants.STREAM_MAGIC),
        highByte(ObjectStreamConstants.STREAM_VERSION),
        lowByte(ObjectStreamConstants.STREAM_VERSION),
        ObjectStreamConstants.TC_ARRAY,
        ObjectStreamConstants.TC_CLASSDESC };

    public static KNNVectorSerializer getSerializerBySerializationMode(final SerializationMode serializationMode) {
        return VECTOR_SERIALIZER_BY_TYPE.getOrDefault(serializationMode, new KNNVectorAsCollectionOfFloatsSerializer());
    }

    public static KNNVectorSerializer getDefaultSerializer() {
        return getSerializerBySerializationMode(COLLECTION_OF_FLOATS);
    }

    public static KNNVectorSerializer getSerializerByStreamContent(final ByteArrayInputStream byteStream) {
        final SerializationMode serializationMode = getSerializerModeFromStream(byteStream);
        return getSerializerBySerializationMode(serializationMode);
    }

    public static SerializationMode getSerializerModeFromStream(ByteArrayInputStream byteStream) {
        int numberOfAvailableBytesInStream = byteStream.available();
        if (numberOfAvailableBytesInStream < ARRAY_HEADER_OFFSET) {
            return getSerializerOrThrowError(numberOfAvailableBytesInStream, COLLECTION_OF_FLOATS);
        }
        final byte[] byteArray = new byte[SERIALIZATION_PROTOCOL_HEADER_PREFIX.length];
        byteStream.read(byteArray, 0, SERIALIZATION_PROTOCOL_HEADER_PREFIX.length);
        byteStream.reset();
        // checking if stream protocol grammar in header is valid for serialized array
        if (Arrays.equals(SERIALIZATION_PROTOCOL_HEADER_PREFIX, byteArray)) {
            int numberOfAvailableBytesAfterHeader = numberOfAvailableBytesInStream - ARRAY_HEADER_OFFSET;
            return getSerializerOrThrowError(numberOfAvailableBytesAfterHeader, ARRAY);
        }
        return getSerializerOrThrowError(numberOfAvailableBytesInStream, COLLECTION_OF_FLOATS);
    }

    private static SerializationMode getSerializerOrThrowError(int numberOfRemainingBytes, final SerializationMode serializationMode) {
        if (numberOfRemainingBytes % BYTES_IN_FLOAT == 0) {
            return serializationMode;
        }
        throw new IllegalArgumentException(
            String.format("Byte stream cannot be deserialized to array of floats due to invalid length %d", numberOfRemainingBytes)
        );
    }

    private static byte highByte(short shortValue) {
        return (byte) (shortValue >> BITS_IN_ONE_BYTE);
    }

    private static byte lowByte(short shortValue) {
        return (byte) shortValue;
    }

}

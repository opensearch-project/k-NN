/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.codec;

import com.google.common.collect.ImmutableMap;

import java.io.ByteArrayInputStream;
import java.io.ObjectStreamConstants;
import java.util.Arrays;
import java.util.Map;

import static org.opensearch.knn.index.codec.SerializationMode.ARRAY;
import static org.opensearch.knn.index.codec.SerializationMode.COLLECTION_OF_FLOATS;

/**
 * Class abstracts Factory for VectorSerializer implementations. Exact implementation constructed and returned based on
 * either content of the byte array or directly based on serialization type.
 */
public class VectorSerializerFactory {
    private static Map<SerializationMode, VectorSerializer> VECTOR_SERIALIZER_BY_TYPE = ImmutableMap.of(
            ARRAY, new VectorAsArraySerializer(),
            COLLECTION_OF_FLOATS, new VectorAsCollectionOfFloatsSerializer()
    );

    private static final int ARRAY_HEADER_OFFSET = 27;

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
            ObjectStreamConstants.TC_CLASSDESC
    };

    public static VectorSerializer getSerializerBySerializationMode(final SerializationMode serializationMode) {
        return VECTOR_SERIALIZER_BY_TYPE.getOrDefault(serializationMode, new VectorAsCollectionOfFloatsSerializer());
    }

    public static VectorSerializer getDefaultSerializer() {
        return getSerializerBySerializationMode(COLLECTION_OF_FLOATS);
    }

    public static VectorSerializer getSerializerByStreamContent(final ByteArrayInputStream byteStream) {
        final SerializationMode serializationMode = serializerModeFromStream(byteStream);
        return getSerializerBySerializationMode(serializationMode);
    }

    private static SerializationMode serializerModeFromStream(ByteArrayInputStream byteStream) {
        //check size, if the length is long enough for header and length is header + some number of floats
        if (byteStream.available() < ARRAY_HEADER_OFFSET ||
                (byteStream.available() - ARRAY_HEADER_OFFSET) % 4 != 0) {
            return COLLECTION_OF_FLOATS;
        }
        final byte[] byteArray = new byte[SERIALIZATION_PROTOCOL_HEADER_PREFIX.length];
        byteStream.read(byteArray, 0, SERIALIZATION_PROTOCOL_HEADER_PREFIX.length);
        byteStream.reset();
        //checking if stream protocol grammar in header is valid for serialized array
        if (Arrays.equals(SERIALIZATION_PROTOCOL_HEADER_PREFIX, byteArray)) {
            return ARRAY;
        }
        return COLLECTION_OF_FLOATS;
    }

    private static byte highByte(short shortValue) {
        return (byte) (shortValue>>8);
    }

    private static byte lowByte(short shortValue) {
        return (byte) shortValue;
    }


}

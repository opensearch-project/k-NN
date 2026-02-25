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

package org.opensearch.knn.jni;

/**
 * Common class for providing the JNI related functionality to various JNIServices.
 */
public class JNICommons {

    static {
        KNNLibraryLoader.loadCommonLibrary();
    }

    /**
     * This is utility function that can be used to store data in native memory. This function will allocate memory for
     * the data(rows*columns) with initialCapacity and return the memory address where the data is stored.
     * If you are using this function for first time use memoryAddress = 0 to ensure that a new memory location is created.
     * For subsequent calls you can pass the same memoryAddress. If the data cannot be stored in the memory location
     * will throw Exception.
     *
     * <p>
     * The function is not threadsafe. If multiple threads are trying to insert on same memory location, then it can
     * lead to data corruption.
     * </p>
     *
     * @param memoryAddress   The address of the memory location where data will be stored.
     * @param data            2D float array containing data to be stored in native memory.
     * @param initialCapacity The initial capacity of the memory location.
     * @return memory address where the data is stored.
     */
    public static long storeVectorData(long memoryAddress, float[][] data, long initialCapacity) {
        return storeVectorData(memoryAddress, data, initialCapacity, true);
    }

    /**
     * This is utility function that can be used to store data in native memory. This function will allocate memory for
     * the data(rows*columns) with initialCapacity and return the memory address where the data is stored.
     * If you are using this function for first time use memoryAddress = 0 to ensure that a new memory location is created.
     * For subsequent calls you can pass the same memoryAddress. If the data cannot be stored in the memory location
     * will throw Exception.
     *
     * <p>
     * The function is not threadsafe. If multiple threads are trying to insert on same memory location, then it can
     * lead to data corruption.
     * </p>
     *
     * @param memoryAddress   The address of the memory location where data will be stored.
     * @param data            2D float array containing data to be stored in native memory.
     * @param initialCapacity The initial capacity of the memory location.
     * @param append          append the data or rewrite the memory location
     * @return memory address where the data is stored.
     */
    public static native long storeVectorData(long memoryAddress, float[][] data, long initialCapacity, boolean append);

    /**
     * This is utility function that can be used to store binary data in native memory. This function will allocate memory for
     * the data(rows*columns) with initialCapacity and return the memory address where the data is stored.
     * If you are using this function for first time use memoryAddress = 0 to ensure that a new memory location is created.
     * For subsequent calls you can pass the same memoryAddress. If the data cannot be stored in the memory location
     * will throw Exception.
     *
     * <p>
     * The function is not threadsafe. If multiple threads are trying to insert on same memory location, then it can
     * lead to data corruption.
     * </p>
     *
     * @param memoryAddress   The address of the memory location where data will be stored.
     * @param data            2D byte array containing binary data to be stored in native memory.
     * @param initialCapacity The initial capacity of the memory location.
     * @return memory address where the data is stored.
     */
    public static long storeBinaryVectorData(long memoryAddress, byte[][] data, long initialCapacity) {
        return storeBinaryVectorData(memoryAddress, data, initialCapacity, true);
    }

    /**
     * This is utility function that can be used to store binary data in native memory. This function will allocate memory for
     * the data(rows*columns) with initialCapacity and return the memory address where the data is stored.
     * If you are using this function for first time use memoryAddress = 0 to ensure that a new memory location is created.
     * For subsequent calls you can pass the same memoryAddress. If the data cannot be stored in the memory location
     * will throw Exception.
     *
     * <p>
     * The function is not threadsafe. If multiple threads are trying to insert on same memory location, then it can
     * lead to data corruption.
     * </p>
     *
     * @param memoryAddress   The address of the memory location where data will be stored.
     * @param data            2D byte array containing binary data to be stored in native memory.
     * @param initialCapacity The initial capacity of the memory location.
     * @param append          append the data or rewrite the memory location
     * @return memory address where the data is stored.
     */
    public static native long storeBinaryVectorData(long memoryAddress, byte[][] data, long initialCapacity, boolean append);

    /**
     * This is utility function that can be used to store byte data in native memory. This function will allocate memory for
     * the data(rows*columns) with initialCapacity and return the memory address where the data is stored.
     * If you are using this function for first time use memoryAddress = 0 to ensure that a new memory location is created.
     * For subsequent calls you can pass the same memoryAddress. If the data cannot be stored in the memory location
     * will throw Exception.
     *
     * <p>
     * The function is not threadsafe. If multiple threads are trying to insert on same memory location, then it can
     * lead to data corruption.
     * </p>
     *
     * @param memoryAddress   The address of the memory location where data will be stored.
     * @param data            2D byte array containing byte data to be stored in native memory.
     * @param initialCapacity The initial capacity of the memory location.
     * @return memory address where the data is stored.
     */
    public static long storeByteVectorData(long memoryAddress, byte[][] data, long initialCapacity) {
        return storeByteVectorData(memoryAddress, data, initialCapacity, true);
    }

    /**
     * This is utility function that can be used to store byte data in native memory. This function will allocate memory for
     * the data(rows*columns) with initialCapacity and return the memory address where the data is stored.
     * If you are using this function for first time use memoryAddress = 0 to ensure that a new memory location is created.
     * For subsequent calls you can pass the same memoryAddress. If the data cannot be stored in the memory location
     * will throw Exception.
     *
     * <p>
     * The function is not threadsafe. If multiple threads are trying to insert on same memory location, then it can
     * lead to data corruption.
     * </p>
     *
     * @param memoryAddress   The address of the memory location where data will be stored.
     * @param data            2D byte array containing byte data to be stored in native memory.
     * @param initialCapacity The initial capacity of the memory location.
     * @param append          append the data or rewrite the memory location
     * @return memory address where the data is stored.
     */
    public static native long storeByteVectorData(long memoryAddress, byte[][] data, long initialCapacity, boolean append);

    /**
     * Free up the memory allocated for the data stored in memory address. This function should be used with the memory
     * address returned by {@link JNICommons#storeVectorData(long, float[][], long, boolean)}
     *
     * <p>
     * The function is not threadsafe. If multiple threads are trying to free up same memory location, then it can
     * lead to errors.
     * </p>
     *
     * @param memoryAddress address to be freed.
     */
    public static native void freeVectorData(long memoryAddress);

    /**
     * Free up the memory allocated for the binary data stored in memory address. This function should be used with the memory
     * address returned by {@link JNICommons#storeBinaryVectorData(long, byte[][], long)}
     *
     * <p>
     * The function is not threadsafe. If multiple threads are trying to free up same memory location, then it can
     * lead to errors.
     * </p>
     *
     * @param memoryAddress address to be freed.
     */
    public static native void freeBinaryVectorData(long memoryAddress);

    /**
     * Free up the memory allocated for the byte data stored in memory address. This function should be used with the memory
     * address returned by {@link JNICommons#storeByteVectorData(long, byte[][], long)}
     *
     * <p>
     *  The function is not threadsafe. If multiple threads are trying to free up same memory location, then it can
     *  lead to errors.
     * </p>
     *
     * @param memoryAddress address to be freed.
     */
    public static native void freeByteVectorData(long memoryAddress);
}

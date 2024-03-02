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

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.util.KNNEngine;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.security.PrivilegedExceptionAction;
import java.util.Map;
import oshi.util.platform.mac.SysctlUtil;

import static org.opensearch.knn.index.KNNSettings.isFaissAVX2Disabled;

/**
 * Service to interact with faiss jni layer. Class dependencies should be minimal
 *
 * In order to compile C++ header file, run:
 * javac -h jni/include src/main/java/org/opensearch/knn/jni/FaissService.java
 *      src/main/java/org/opensearch/knn/index/query/KNNQueryResult.java
 *      src/main/java/org/opensearch/knn/common/KNNConstants.java
 */
class FaissService {
    private static Logger logger = LogManager.getLogger(FaissService.class);

    static {
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {

            // Even if the underlying system supports AVX2, users can override and disable it by using the
            // 'knn.faiss.avx2.disabled' setting by setting it to true in the opensearch.yml configuration
            if (!isFaissAVX2Disabled() && isAVX2SupportedBySystem()) {
                System.loadLibrary(KNNConstants.FAISS_AVX2_JNI_LIBRARY_NAME);
            } else {
                System.loadLibrary(KNNConstants.FAISS_JNI_LIBRARY_NAME);
            }

            initLibrary();
            KNNEngine.FAISS.setInitialized(true);
            return null;
        });
    }

    /**
     * Verify if the underlying system supports AVX2 SIMD Optimization or not
     * 1. If the architecture is aarch64 return false.
     * 2. If the operating system is windows return false.
     * 3. If the operating system is macOS, use oshi-core library to verify if the cpu flags
     *    contains 'avx2' and return true if it exists else false.
     * 4. If the operating system is linux, read the '/proc/cpuinfo' file path and verify if
     *    the flags contains 'avx2' and return true if it exists else false.
     */
    private static boolean isAVX2SupportedBySystem() {
        if ((System.getProperty("os.arch").toLowerCase()).contains("aarch")
            || (System.getProperty("os.name").toLowerCase()).contains("windows")) {
            return false;
        }

        if ((System.getProperty("os.name").toLowerCase()).contains("mac")) {
            try {
                return AccessController.doPrivileged((PrivilegedExceptionAction<Boolean>) () -> {
                    String flags = SysctlUtil.sysctl("machdep.cpu.leaf7_features", "empty");
                    if ((flags.toLowerCase()).contains("avx2")) {
                        return true;
                    }
                    return false;
                });
            } catch (Exception e) {
                logger.error("[KNN] Error fetching cpu flags info. [{}]", e.getMessage());
                e.printStackTrace();
            }

        } else if ((System.getProperty("os.name").toLowerCase()).contains("linux")) {
            String fileName = "/proc/cpuinfo";
            try {
                return AccessController.doPrivileged((PrivilegedExceptionAction<Boolean>) () -> {
                    String cpuFlags = Files.lines(Paths.get(fileName))
                        .filter(s -> s.startsWith("flags"))
                        .filter(s -> s.contains("avx2"))
                        .findFirst()
                        .orElse("");

                    if ((cpuFlags.toLowerCase()).contains("avx2")) {
                        return true;
                    }
                    return false;
                });

            } catch (Exception e) {
                logger.error("[KNN] Error reading file [{}]. [{}]", fileName, e.getMessage());
                e.printStackTrace();
            }
        }
        return false;
    }

    /**
     * Create an index for the native library
     *
     * @param ids array of ids mapping to the data passed in
     * @param data array of float arrays to be indexed
     * @param indexPath path to save index file to
     * @param parameters parameters to build index
     */
    public static native void createIndex(int[] ids, float[][] data, String indexPath, Map<String, Object> parameters);

    /**
     * Create an index for the native library with a provided template index
     *
     * @param ids array of ids mapping to the data passed in
     * @param data array of float arrays to be indexed
     * @param indexPath path to save index file to
     * @param templateIndex empty template index
     * @param parameters additional build time parameters
     */
    public static native void createIndexFromTemplate(
        int[] ids,
        float[][] data,
        String indexPath,
        byte[] templateIndex,
        Map<String, Object> parameters
    );

    /**
     * Load an index into memory
     *
     * @param indexPath path to index file
     * @return pointer to location in memory the index resides in
     */
    public static native long loadIndex(String indexPath);

    /**
     * Query an index without filter
     *
     * If the "knn" field is a nested field, each vector value within that nested field will be assigned its
     * own document ID. In this situation, the term "parent ID" corresponds to the original document ID.
     * The arrangement of parent IDs and nested field IDs is assured to have all nested field IDs appearing first,
     * followed by the parent ID, in consecutive order without any gaps. Because of this ID pattern,
     * we can determine the parent ID of a specific nested field ID using only an array of parent IDs.
     *
     * @param indexPointer pointer to index in memory
     * @param queryVector vector to be used for query
     * @param k neighbors to be returned
     * @param parentIds list of parent doc ids when the knn field is a nested field
     * @return KNNQueryResult array of k neighbors
     */
    public static native KNNQueryResult[] queryIndex(long indexPointer, float[] queryVector, int k, int[] parentIds);

    /**
     * Query an index with filter
     *
     * @param indexPointer pointer to index in memory
     * @param queryVector vector to be used for query
     * @param k neighbors to be returned
     * @param filterIds list of doc ids to include in the query result
     * @param parentIds list of parent doc ids when the knn field is a nested field
     * @return KNNQueryResult array of k neighbors
     */
    public static native KNNQueryResult[] queryIndexWithFilter(
        long indexPointer,
        float[] queryVector,
        int k,
        long[] filterIds,
        int filterIdsType,
        int[] parentIds
    );

    /**
     * Free native memory pointer
     */
    public static native void free(long indexPointer);

    /**
     * Initialize library
     *
     */
    public static native void initLibrary();

    /**
     * Train an empty index
     *
     * @param indexParameters parameters used to build index
     * @param dimension dimension for the index
     * @param trainVectorsPointer pointer to where training vectors are stored in native memory
     * @return bytes array of trained template index
     */
    public static native byte[] trainIndex(Map<String, Object> indexParameters, int dimension, long trainVectorsPointer);

    /**
     * Transfer vectors from Java to native
     *
     * @param vectorsPointer pointer to vectors in native memory. Should be 0 to create vector as well
     * @param trainingData data to be transferred
     * @return pointer to native memory location of training data
     */
    public static native long transferVectors(long vectorsPointer, float[][] trainingData);

    /**
     * Free vectors from memory
     *
     * @param vectorsPointer to be freed
     */
    public static native void freeVectors(long vectorsPointer);
}

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

package org.opensearch.knn;

import org.opensearch.common.xcontent.DeprecationHandler;
import org.opensearch.common.xcontent.NamedXContentRegistry;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class TestUtils {
    public static final String KNN_BWC_PREFIX = "knn-bwc-";
    public static final String OS_KNN = "opensearch-knn";
    public static final String OPENDISTRO_SECURITY = ".opendistro_security";
    public static final String BWCSUITE_CLUSTER = "tests.rest.bwcsuite_cluster";
    public static final String BWCSUITE_ROUND = "tests.rest.bwcsuite_round";
    public static final String TEST_CLUSTER_NAME = "tests.clustername";

    /**
     * Class to read in some test data from text files
     */
    public static class TestData {
        public KNNCodecUtil.Pair indexData;
        public float[][] queries;

        public TestData(String testIndexVectorsPath, String testQueriesPath) throws IOException {
            indexData = readIndexData(testIndexVectorsPath);
            queries = readQueries(testQueriesPath);
        }

        private KNNCodecUtil.Pair readIndexData(String path) throws IOException {
            List<Integer> idsList = new ArrayList<>();
            List<Float[]> vectorsList = new ArrayList<>();

            BufferedReader reader = new BufferedReader(new FileReader(path));
            String line = reader.readLine();
            while (line != null) {
                Map<String, Object> doc = XContentFactory.xContent(XContentType.JSON).createParser(
                        NamedXContentRegistry.EMPTY, DeprecationHandler.THROW_UNSUPPORTED_OPERATION, line).map();
                idsList.add((Integer) doc.get("id"));

                @SuppressWarnings("unchecked")
                ArrayList<Double> vector = (ArrayList<Double>) doc.get("vector");
                Float[] floatArray = new Float[vector.size()];
                for (int i =0; i< vector.size(); i++) {
                    floatArray[i] = vector.get(i).floatValue();
                }
                vectorsList.add(floatArray);

                line = reader.readLine();
            }
            reader.close();

            int[] idsArray = new int [idsList.size()];
            float[][] vectorsArray = new float[vectorsList.size()][vectorsList.get(0).length];
            for (int i = 0; i < idsList.size(); i++) {
                idsArray[i] = idsList.get(i);

                for (int j = 0; j < vectorsList.get(i).length; j++) {
                    vectorsArray[i][j] = vectorsList.get(i)[j];
                }
            }

            return new KNNCodecUtil.Pair(idsArray, vectorsArray);
        }

        private float[][] readQueries(String path) throws IOException {
            BufferedReader reader = new BufferedReader(new FileReader(path));
            String line = reader.readLine();

            List<Float[]> floatsList = new ArrayList<>();

            while (line != null) {
                String[] floatStrings = line.split(",");

                Float[] queryArray = new Float[floatStrings.length];
                for (int i = 0; i < queryArray.length; i++) {
                    queryArray[i] = Float.parseFloat(floatStrings[i]);
                }

                floatsList.add(queryArray);

                line = reader.readLine();
            }
            reader.close();

            float[][] queryArray = new float[floatsList.size()][floatsList.get(0).length];
            for (int i = 0; i < queryArray.length; i++) {
                for (int j = 0; j < queryArray[i].length; j++) {
                    queryArray[i][j] = floatsList.get(i)[j];
                }
            }
            return queryArray;
        }
    }
}

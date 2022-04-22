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
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.plugin.script.KNNScoringUtil;
import java.util.Comparator;
import java.util.Random;
import java.util.Set;
import java.util.PriorityQueue;
import java.util.ArrayList;
import java.util.List;
import java.util.HashSet;
import java.util.Map;

import static org.apache.lucene.tests.util.LuceneTestCase.random;

class DistVector {
    public float dist;
    public String docID;

    public DistVector(float dist, String docID) {
        this.dist = dist;
        this.docID = docID;
    }

    public String getDocID() {
        return docID;
    }

    public float getDist() {
        return dist;
    }
}

class DistComparator implements Comparator<DistVector> {

    public int compare(DistVector d1, DistVector d2) {
        if (d1.dist < d2.dist) {
            return 1;
        } else if (d1.dist > d2.dist) {
            return -1;
        }
        return 0;
    }
}

public class TestUtils {
    public static final String KNN_BWC_PREFIX = "knn-bwc-";
    public static final String OS_KNN = "opensearch-knn";
    public static final String OPENDISTRO_SECURITY = ".opendistro_security";
    public static final String BWCSUITE_CLUSTER = "tests.rest.bwcsuite_cluster";
    public static final String BWCSUITE_ROUND = "tests.rest.bwcsuite_round";
    public static final String BWC_VERSION = "tests.plugin_bwc_version";
    public static final String CLIENT_TIMEOUT_VALUE = "90s";
    public static final String MIXED_CLUSTER = "mixed_cluster";
    public static final String NODES_BWC_CLUSTER = "3";
    public static final String OLD_CLUSTER = "old_cluster";
    public static final String RESTART_UPGRADE_OLD_CLUSTER = "tests.is_old_cluster";
    public static final String ROLLING_UPGRADE_FIRST_ROUND = "tests.rest.first_round";
    public static final String TEST_CLUSTER_NAME = "tests.clustername";
    public static final String UPGRADED_CLUSTER = "upgraded_cluster";

    // Generating vectors using random function with a seed which makes these vectors standard and generate same vectors for each run.
    public static float[][] randomlyGenerateStandardVectors(int numVectors, int dimensions, int seed) {
        float[][] standardVectors = new float[numVectors][dimensions];
        Random rand = new Random(seed);

        for (int i = 0; i < numVectors; i++) {
            float[] vector = new float[dimensions];
            for (int j = 0; j < dimensions; j++) {
                vector[j] = rand.nextFloat();
            }
            standardVectors[i] = vector;
        }
        return standardVectors;
    }

    public static float[][] generateRandomVectors(int numVectors, int dimensions) {
        float[][] randomVectors = new float[numVectors][dimensions];

        for (int i = 0; i < numVectors; i++) {
            float[] vector = new float[dimensions];
            for (int j = 0; j < dimensions; j++) {
                vector[j] = random().nextFloat();
            }
            randomVectors[i] = vector;
        }
        return randomVectors;
    }

    /*
     * Here, for a given space type we will compute the 'k' shortest distances among all the index vectors for each and every query vector using a priority queue and
     * their document ids are stored. These document ids are later used while calculating Recall value to compare with the document ids of 'k' results obtained for
     * each and every search query performed.
     */
    public static List<Set<String>> computeGroundTruthValues(float[][] indexVectors, float[][] queryVectors, SpaceType spaceType, int k) {
        ArrayList<Set<String>> groundTruthValues = new ArrayList<>();
        PriorityQueue<DistVector> pq;
        HashSet<String> docIds;
        float dist = 0.0f;

        for (int i = 0; i < queryVectors.length; i++) {
            pq = new PriorityQueue<>(k, new DistComparator());
            for (int j = 0; j < indexVectors.length; j++) {
                if (spaceType != null && "l2".equals(spaceType.getValue())) {
                    dist = KNNScoringUtil.l2Squared(queryVectors[i], indexVectors[j]);
                }

                if (pq.size() < k) {
                    pq.add(new DistVector(dist, String.valueOf(j + 1)));
                } else if (pq.peek().getDist() > dist) {
                    pq.poll();
                    pq.add(new DistVector(dist, String.valueOf(j + 1)));
                }
            }

            docIds = new HashSet<>();
            while (!pq.isEmpty()) {
                docIds.add(pq.poll().getDocID());
            }

            groundTruthValues.add(docIds);
        }

        return groundTruthValues;
    }

    public static float[][] getQueryVectors(int queryCount, int dimensions, int docCount, boolean isStandard) {
        if (isStandard) {
            return randomlyGenerateStandardVectors(queryCount, dimensions, docCount + 1);
        } else {
            return generateRandomVectors(queryCount, dimensions);
        }
    }

    public static float[][] getIndexVectors(int docCount, int dimensions, boolean isStandard) {
        if (isStandard) {
            return randomlyGenerateStandardVectors(docCount, dimensions, 1);
        } else {
            return generateRandomVectors(docCount, dimensions);
        }
    }

    /*
     * Recall is the number of relevant documents retrieved by a search divided by the total number of existing relevant documents.
     * We are similarly calculating recall by verifying number of relevant documents obtained in the search results by comparing with
     * groundTruthValues and then dividing by 'k'
     */
    public static double calculateRecallValue(List<List<String>> searchResults, List<Set<String>> groundTruthValues, int k) {
        ArrayList<Float> recalls = new ArrayList<>();

        for (int i = 0; i < searchResults.size(); i++) {
            float recallVal = 0.0F;
            for (int j = 0; j < searchResults.get(i).size(); j++) {
                if (groundTruthValues.get(i).contains(searchResults.get(i).get(j))) {
                    recallVal += 1.0;
                }
            }
            recalls.add(recallVal / k);
        }

        double sum = recalls.stream().reduce((a, b) -> a + b).get();
        return sum / recalls.size();
    }

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
                Map<String, Object> doc = XContentFactory.xContent(XContentType.JSON)
                    .createParser(NamedXContentRegistry.EMPTY, DeprecationHandler.THROW_UNSUPPORTED_OPERATION, line)
                    .map();
                idsList.add((Integer) doc.get("id"));

                @SuppressWarnings("unchecked")
                ArrayList<Double> vector = (ArrayList<Double>) doc.get("vector");
                Float[] floatArray = new Float[vector.size()];
                for (int i = 0; i < vector.size(); i++) {
                    floatArray[i] = vector.get(i).floatValue();
                }
                vectorsList.add(floatArray);

                line = reader.readLine();
            }
            reader.close();

            int[] idsArray = new int[idsList.size()];
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

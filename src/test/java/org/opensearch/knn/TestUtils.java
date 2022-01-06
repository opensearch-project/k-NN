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
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.plugin.script.KNNScoringUtil;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Arrays;
import static org.apache.lucene.util.LuceneTestCase.random;


public class TestUtils {
    public static final String KNN_BWC_PREFIX = "knn-bwc-";
    public static final String OS_KNN = "opensearch-knn";
    public static final String OPENDISTRO_SECURITY = ".opendistro_security";
    public static final String BWCSUITE_CLUSTER = "tests.rest.bwcsuite_cluster";
    public static final String BWCSUITE_ROUND = "tests.rest.bwcsuite_round";
    public static final String TEST_CLUSTER_NAME = "tests.clustername";

    //Generating index vectors using random function with a seed which makes these vectors standard and generate same vectors for each run.
    public static float[][] randomlyGenerateStandardIndexVectors(int numVectors, int dimensions) {
        float[][] standardIndexVectors = new float[numVectors][dimensions];
        int seedVal = 500;

        for (int i = 0; i < numVectors; i++) {
            float[] vector = new float[dimensions];
            for (int j = 0; j < dimensions; j++) {
                Random random = new Random(seedVal++);
                vector[j] = random.nextFloat();
            }
            standardIndexVectors[i] = vector;
        }
        return standardIndexVectors;
    }

    //Generating query vectors using random function with a seed which makes these vectors standard and generate same vectors for each run.
    public static float[][] randomlyGenerateStandardQueryVectors(int numVectors, int dimensions){
        float[][] standardQueryVectors = new float[numVectors][dimensions];
        int seedVal = 1;

        for (int i = 0; i < numVectors; i++) {
            float[] vector = new float[dimensions];
            for (int j = 0; j < dimensions; j++) {
                Random random = new Random(seedVal++);
                vector[j] = random.nextFloat();
            }
            standardQueryVectors[i] = vector;
        }
        return standardQueryVectors;
    }

    public static float[][] generateRandomVectors(int numVectors, int dimensions){
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

    // Here, for a given space type we will compute the 'k' shortest distances among all the index vectors for each and every query vector.
    // These computed distances are later used while calculating Recall value to compare with the 'k' index vectors obtained for each and every search query performed.
    public static float[][] computeGroundTruthDistances(float[][] indexVectors, float[][] queryVectors, SpaceType spaceType, int k){
        float[][] groundTruthValues = new float[queryVectors.length][k];
        for(int i = 0; i < queryVectors.length; i++){
            float[] distValues = new float[indexVectors.length];
            for(int j = 0; j < indexVectors.length; j++){
                if(spaceType != null && "l2".equals(spaceType.getValue()))
                    distValues[j] = KNNScoringUtil.l2Squared(queryVectors[i], indexVectors[j]);
            }
            Arrays.sort(distValues);
            System.arraycopy(distValues, 0, groundTruthValues[i], 0, k);
        }
        return groundTruthValues;
    }

    public static float[][] getQueryVectors(int queryCount, int dimensions, boolean isStandard){
        if(isStandard)
            return randomlyGenerateStandardQueryVectors(queryCount, dimensions);
        else
            return generateRandomVectors(queryCount, dimensions);
    }

    public static float[][] getIndexVectors(int docCount, int dimensions, boolean isStandard){
        if(isStandard)
            return randomlyGenerateStandardIndexVectors(docCount, dimensions);
        else
            return generateRandomVectors(docCount, dimensions);
    }

    //Computing recall value by computing distances for the obtained query vectors and search results based on the given spaceType and comparing those distances with the
    //computed ground truth values
    public static double calculateRecallValue(float[][] queryVectors, List<List<float[]>> searchResults, float[][] groundTruthValues, int k, SpaceType spaceType){
        ArrayList<Float> recalls = new ArrayList<>();
        float dist = 0.0F;

        for(int i = 0; i < queryVectors.length; i++){
            float recallVal = 0.0F;
            for(int j = 0; j < searchResults.get(i).size(); j++){
                if(spaceType != null && "l2".equals(spaceType.getValue())){
                    dist = KNNScoringUtil.l2Squared(queryVectors[i], searchResults.get(i).get(j));
                }

                if (dist <= groundTruthValues[i][j]){
                    recallVal += 1.0;
                }
            }
            recalls.add(recallVal / k);
        }

        double sum = recalls.stream().reduce((a,b)->a+b).get();
        return sum/recalls.size();
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

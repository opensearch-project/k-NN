/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import com.google.common.collect.ImmutableMap;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.util.SerializationMode;
import org.opensearch.knn.jni.JNICommons;
import org.opensearch.knn.plugin.script.KNNScoringUtil;

import java.util.Collections;
import java.util.Comparator;
import java.util.Random;
import java.util.Set;
import java.util.PriorityQueue;
import java.util.ArrayList;
import java.util.List;
import java.util.HashSet;
import java.util.Map;
import java.util.function.BiFunction;

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

    public static Map<SpaceType, BiFunction<float[], float[], Float>> KNN_SCORING_SPACE_TYPE = ImmutableMap.of(
        SpaceType.L1,
        KNNScoringUtil::l1Norm,
        SpaceType.L2,
        KNNScoringUtil::l2Squared,
        SpaceType.LINF,
        KNNScoringUtil::lInfNorm,
        SpaceType.COSINESIMIL,
        KNNScoringUtil::cosinesimil,
        SpaceType.INNER_PRODUCT,
        KNNScoringUtil::innerProduct
    );

    public static final String KNN_BWC_PREFIX = "knn-bwc-";
    public static final String OPENDISTRO_SECURITY = ".opendistro_security";
    public static final String BWCSUITE_CLUSTER = "tests.rest.bwcsuite_cluster";
    public static final String BWC_VERSION = "tests.plugin_bwc_version";
    public static final String CLIENT_TIMEOUT_VALUE = "90s";
    public static final String FIELD = "field";
    public static final int KNN_ALGO_PARAM_M_MIN_VALUE = 2;
    public static final int KNN_ALGO_PARAM_EF_CONSTRUCTION_MIN_VALUE = 2;
    public static final String MIXED_CLUSTER = "mixed_cluster";
    public static final String NODES_BWC_CLUSTER = "3";
    public static final String NUMBER_OF_SHARDS = "number_of_shards";
    public static final String NUMBER_OF_REPLICAS = "number_of_replicas";
    public static final String INDEX_KNN = "index.knn";
    public static final String OLD_CLUSTER = "old_cluster";
    public static final String PROPERTIES = "properties";
    public static final String VECTOR_TYPE = "type";
    public static final String KNN_VECTOR = "knn_vector";
    public static final String QUERY_VALUE = "query_value";
    public static final String RESTART_UPGRADE_OLD_CLUSTER = "tests.is_old_cluster";
    public static final String ROLLING_UPGRADE_FIRST_ROUND = "tests.rest.first_round";
    public static final String SKIP_DELETE_MODEL_INDEX = "tests.skip_delete_model_index";
    public static final String UPGRADED_CLUSTER = "upgraded_cluster";
    public static final String SECURITY_AUDITLOG_PREFIX = "security-auditlog";
    public static final String OPENSEARCH_SYSTEM_INDEX_PREFIX = ".opensearch";
    public static final String ML_PLUGIN_SYSTEM_INDEX_PREFIX = ".plugins-ml";

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

        for (int i = 0; i < queryVectors.length; i++) {
            pq = new PriorityQueue<>(k, new DistComparator());
            for (int j = 0; j < indexVectors.length; j++) {
                float dist = computeDistFromSpaceType(spaceType, indexVectors[j], queryVectors[i]);

                // Need to invert distance for IP or COSINE because higher is better in these cases
                if (spaceType == SpaceType.INNER_PRODUCT || spaceType == SpaceType.COSINESIMIL) {
                    dist *= -1;
                }

                pq = insertWithOverflow(pq, k, dist, j);
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

    public static PriorityQueue<DistVector> computeGroundTruthValues(int k, SpaceType spaceType, IDVectorProducer idVectorProducer) {
        PriorityQueue<DistVector> pq = new PriorityQueue<>(k, new DistComparator());
        int numDocs = idVectorProducer.getVectorCount();
        float[] queryVector = idVectorProducer.getVector(numDocs);

        for (int id = 0; id < numDocs; id++) {
            float[] indexVector = idVectorProducer.getVector(id);
            float dist = computeDistFromSpaceType(spaceType, indexVector, queryVector);
            // Need to invert distance for IP or COSINE because higher is better in these cases
            if (spaceType == SpaceType.INNER_PRODUCT || spaceType == SpaceType.COSINESIMIL) {
                dist *= -1;
            }

            pq = insertWithOverflow(pq, k, dist, id);
        }
        return pq;
    }

    public static float computeDistFromSpaceType(SpaceType spaceType, float[] indexVector, float[] queryVector) {
        float dist;
        if (spaceType != null) {
            dist = KNN_SCORING_SPACE_TYPE.getOrDefault(spaceType, (defaultQueryVector, defaultIndexVector) -> {
                throw new IllegalArgumentException(String.format("Invalid SpaceType function: \"%s\"", spaceType));
            }).apply(queryVector, indexVector);
        } else {
            throw new NullPointerException("SpaceType is null. Provide a valid SpaceType.");
        }
        return dist;
    }

    // If the priority queue size is less than k, it adds a new DistVector(distance and vector id)
    // If the priority queue size is full, then it compares the distance and replaces the top element
    // with new DistVector if new dist is less than the dist of top element
    public static PriorityQueue<DistVector> insertWithOverflow(PriorityQueue<DistVector> pq, int k, float dist, int id) {
        if (pq.size() < k) {
            pq.add(new DistVector(dist, String.valueOf(id)));
        } else if (pq.peek().getDist() > dist) {
            pq.poll();
            pq.add(new DistVector(dist, String.valueOf(id)));
        }
        return pq;
    }

    /**
     * Class to read in some test data from text files
     */
    public static class TestData {
        public Pair indexData;
        public byte[][] indexBinaryData;
        public float[][] queries;
        public String[][] groundTruthValues;
        public byte[][] binaryQueries;

        public TestData(String testIndexVectorsPath, String testQueriesPath) throws IOException {
            indexData = readIndexData(testIndexVectorsPath);
            queries = readQueries(testQueriesPath);
            initBinaryData();
        }

        public TestData(String testIndexVectorsPath, String testQueriesPath, String groundTruthValuesPath) throws IOException {
            indexData = readIndexData(testIndexVectorsPath);
            queries = readQueries(testQueriesPath);
            groundTruthValues = readGroundTruthValues(groundTruthValuesPath);
            initBinaryData();
        }

        private Pair readIndexData(String path) throws IOException {
            List<Integer> idsList = new ArrayList<>();
            List<Float[]> vectorsList = new ArrayList<>();

            BufferedReader reader = new BufferedReader(new FileReader(path));
            String line = reader.readLine();
            while (line != null) {
                Map<String, Object> doc = XContentHelper.createParser(
                    NamedXContentRegistry.EMPTY,
                    DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                    new BytesArray(line),
                    MediaTypeRegistry.getDefaultMediaType()
                ).map();
                idsList.add((Integer) doc.get("id"));

                @SuppressWarnings("unchecked")
                ArrayList<Object> vector = (ArrayList<Object>) doc.get("vector");
                Float[] floatArray = new Float[vector.size()];
                for (int i = 0; i < vector.size(); i++) {
                    floatArray[i] = Float.valueOf(vector.get(i).toString());
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
            return new Pair(idsArray, vectorsArray[0].length, SerializationMode.COLLECTION_OF_FLOATS, vectorsArray);
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

        private String[][] readGroundTruthValues(String path) throws IOException {
            BufferedReader reader = new BufferedReader(new FileReader(path));
            String line = reader.readLine();

            List<String[]> stringList = new ArrayList<>();

            while (line != null) {
                String[] intStrings = line.split(",");
                stringList.add(intStrings);
                line = reader.readLine();
            }
            reader.close();

            String[][] docIdArray = new String[stringList.size()][stringList.get(0).length];
            for (int i = 0; i < docIdArray.length; i++) {
                for (int j = 0; j < docIdArray[i].length; j++) {
                    docIdArray[i][j] = stringList.get(i)[j].trim();
                }
            }
            return docIdArray;
        }

        private void initBinaryData() {
            // Find medium value
            List<Float> flattenedVectors = new ArrayList<>(indexData.vectors.length * indexData.vectors[0].length);
            for (int i = 0; i < indexData.vectors.length; i++) {
                for (int j = 0; j < indexData.vectors[i].length; j++) {
                    flattenedVectors.add(indexData.vectors[i][j]);
                }
            }
            Collections.sort(flattenedVectors);
            Float median = flattenedVectors.get(flattenedVectors.size() / 2);

            // Quantize(indexData.vectors[i][j] >= median ? 1 : 0) and
            // packing(8 bits to 1 byte) for index data
            indexBinaryData = new byte[indexData.vectors.length][(indexData.vectors[0].length + 7) / 8];
            for (int i = 0; i < indexData.vectors.length; i++) {
                for (int j = 0; j < indexData.vectors[i].length; j++) {
                    int byteIndex = j / 8;
                    int bitIndex = 7 - (j % 8);
                    indexBinaryData[i][byteIndex] |= (indexData.vectors[i][j] >= median ? 1 : 0) << bitIndex;
                }
            }

            // Quantize(queries[i][j] >= median ? 1 : 0) and
            // packing(8 bits to 1 byte) for query data
            binaryQueries = new byte[queries.length][(queries[0].length + 7) / 8];
            for (int i = 0; i < queries.length; i++) {
                for (int j = 0; j < queries[i].length; j++) {
                    int byteIndex = j / 8;
                    int bitIndex = 7 - (j % 8);
                    binaryQueries[i][byteIndex] |= (queries[i][j] >= median ? 1 : 0) << bitIndex;
                }
            }
        }

        public long loadDataToMemoryAddress() {
            return JNICommons.storeVectorData(0, indexData.vectors, (long) indexData.vectors.length * indexData.vectors[0].length);
        }

        public long loadBinaryDataToMemoryAddress() {
            return JNICommons.storeByteVectorData(0, indexBinaryData, (long) indexBinaryData.length * indexBinaryData[0].length);
        }

        @AllArgsConstructor
        public static class Pair {
            public int[] docs;
            @Getter
            @Setter
            private int dimension;
            public SerializationMode serializationMode;
            public float[][] vectors;
        }
    }
}

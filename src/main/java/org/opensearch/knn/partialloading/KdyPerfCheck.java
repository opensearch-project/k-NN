/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading;

public class KdyPerfCheck {
    private static boolean TURN_ON = false;
    private static ThreadLocal<KdyPerfCheck> THREAD_LOCAL = ThreadLocal.withInitial(KdyPerfCheck::new);

    private long startTime;
    private long parameterPrepareDoneTime;
    private long faissSearchPrepareDoneTime;
    private long faissSearchDoneTime;
    private long queryResultsDoneTime;

    // Id map
    private long faissIdMapStartTime;
    private long faissIdMapNestedSearchStartTime;
    private long faissIdMapIdTransformDoneTime;
    private long faissIdMapDoneTime;
    private long faissIdMapNestedSearchDoneTime;

    // Index HNSW flat
    private long indexHnswFlatStartTime;
    private long hnswSearchStartTime;
    private long hnswSearchDoneTime;
    private long indexHnswFlatDoneTime;

    // HNSW
    private long entryPointAfterGreedy;
    private float entryPointAfterGreedyDistance;
    private long faissHnswStartTime;
    private long hnswGreedyDoneTime;
    private long hnswBfsDoneTime;
    private long faissHnswDoneTime;
    private int hnswEntryPoint;
    private int numVisits;
    private int numVisitsDuringGreedy;

    public static void start() {
        if (TURN_ON) {
            KdyPerfCheck perf = THREAD_LOCAL.get();
            perf.startTime = System.nanoTime();

            perf.parameterPrepareDoneTime = -1;
            perf.faissSearchPrepareDoneTime = -1;
            perf.faissSearchDoneTime = -1;
            perf.queryResultsDoneTime = -1;

            perf.faissIdMapStartTime = -1;
            perf.faissIdMapNestedSearchStartTime = -1;
            perf.faissIdMapIdTransformDoneTime = -1;
            perf.faissIdMapDoneTime = -1;
            perf.faissIdMapNestedSearchDoneTime = -1;

            perf.indexHnswFlatStartTime = -1;
            perf.hnswSearchStartTime = -1;
            perf.hnswSearchDoneTime = -1;
            perf.indexHnswFlatDoneTime = -1;

            perf.entryPointAfterGreedy = -1;
            perf.entryPointAfterGreedyDistance = -1;
            perf.faissHnswStartTime = -1;
            perf.hnswGreedyDoneTime = -1;
            perf.hnswBfsDoneTime = -1;
            perf.faissHnswDoneTime = -1;
            perf.numVisits = 0;
            perf.numVisitsDuringGreedy = 0;
        }
    }

    public static void parameterPrepareDone() {
        if (TURN_ON) {
            KdyPerfCheck perf = THREAD_LOCAL.get();
            perf.parameterPrepareDoneTime = System.nanoTime();
        }
    }

    public static void faissSearchPrepareDone() {
        if (TURN_ON) {
            KdyPerfCheck perf = THREAD_LOCAL.get();
            perf.faissSearchPrepareDoneTime = System.nanoTime();
        }
    }

    public static void faissSearchDone() {
        if (TURN_ON) {
            KdyPerfCheck perf = THREAD_LOCAL.get();
            perf.faissSearchDoneTime = System.nanoTime();
        }
    }

    public static void queryResultsDone() {
        if (TURN_ON) {
            KdyPerfCheck perf = THREAD_LOCAL.get();
            perf.queryResultsDoneTime = System.nanoTime();
        }
    }

    //
    // Id map
    //
    public static void faissIdMapNestedSearchStart() {
        if (TURN_ON) {
            KdyPerfCheck perf = THREAD_LOCAL.get();
            perf.faissIdMapNestedSearchStartTime = System.nanoTime();
        }
    }

    public static void faissIdMapNestedSearchDone() {
        if (TURN_ON) {
            KdyPerfCheck perf = THREAD_LOCAL.get();
            perf.faissIdMapNestedSearchDoneTime = System.nanoTime();
        }
    }

    public static void faissIdMapIdTransformDone() {
        if (TURN_ON) {
            KdyPerfCheck perf = THREAD_LOCAL.get();
            perf.faissIdMapIdTransformDoneTime = System.nanoTime();
        }
    }

    public static void faissIdMapStart() {
        if (TURN_ON) {
            KdyPerfCheck perf = THREAD_LOCAL.get();
            perf.faissIdMapStartTime = System.nanoTime();
        }
    }

    public static void faissIdMapDone() {
        if (TURN_ON) {
            KdyPerfCheck perf = THREAD_LOCAL.get();
            perf.faissIdMapDoneTime = System.nanoTime();
        }
    }

    //
    // Index HNSW flat
    //

    public static void indexHnswFlatStart() {
        if (TURN_ON) {
            KdyPerfCheck perf = THREAD_LOCAL.get();
            perf.indexHnswFlatStartTime = System.nanoTime();
        }
    }

    public static void hnswSearchStart() {
        if (TURN_ON) {
            KdyPerfCheck perf = THREAD_LOCAL.get();
            perf.hnswSearchStartTime = System.nanoTime();
        }
    }

    public static void hnswSearchDone() {
        if (TURN_ON) {
            KdyPerfCheck perf = THREAD_LOCAL.get();
            perf.hnswSearchDoneTime = System.nanoTime();
        }
    }

    public static void indexHnswFlatDone() {
        if (TURN_ON) {
            KdyPerfCheck perf = THREAD_LOCAL.get();
            perf.indexHnswFlatDoneTime = System.nanoTime();
        }
    }

    //
    // HNSW
    //

    public static void faissHnswStart() {
        if (TURN_ON) {
            KdyPerfCheck perf = THREAD_LOCAL.get();
            perf.faissHnswStartTime = System.nanoTime();
        }
    }

    public static void hnswGreedyDone() {
        if (TURN_ON) {
            KdyPerfCheck perf = THREAD_LOCAL.get();
            perf.hnswGreedyDoneTime = System.nanoTime();
            perf.numVisitsDuringGreedy = perf.numVisits;
        }
    }

    public static void hnswBfsDone() {
        if (TURN_ON) {
            KdyPerfCheck perf = THREAD_LOCAL.get();
            perf.hnswBfsDoneTime = System.nanoTime();
        }
    }

    public static void faissHnswDone() {
        if (TURN_ON) {
            KdyPerfCheck perf = THREAD_LOCAL.get();
            perf.faissHnswDoneTime = System.nanoTime();
        }
    }

    public static void setEntryPointAfterGreedy(long entryPoint, float distance) {
        if (TURN_ON) {
            KdyPerfCheck perf = THREAD_LOCAL.get();
            perf.entryPointAfterGreedy = entryPoint;
            perf.entryPointAfterGreedyDistance = distance;
        }
    }

    public synchronized static void report() {
        if (TURN_ON) {
            KdyPerfCheck perf = THREAD_LOCAL.get();
            System.out.println("\n\n================ KDY PERF ======================");
            System.out.println("Total execution time: " + (perf.queryResultsDoneTime - perf.startTime));
            System.out.println("Search parameter preparation time: " + (perf.parameterPrepareDoneTime - perf.startTime));
            System.out.println("FAISS search preparation time: " + (perf.faissSearchPrepareDoneTime - perf.parameterPrepareDoneTime));
            System.out.println("FAISS search time: " + (perf.faissSearchDoneTime - perf.faissSearchPrepareDoneTime));
            System.out.println("FAISS results build time: " + (perf.queryResultsDoneTime - perf.faissSearchDoneTime));

            //
            // Id map
            //
            System.out.println("# Id map index");
            System.out.println("  Id map index total execution time: " + (perf.faissIdMapDoneTime - perf.faissIdMapStartTime));
            System.out.println(
                "  Id map index selector, grouper setting time: " + (perf.faissIdMapNestedSearchStartTime - perf.faissIdMapStartTime));
            System.out.println("  Id map index nested index search time: " + (perf.faissIdMapNestedSearchDoneTime
                - perf.faissIdMapNestedSearchStartTime));
            System.out.println("  Id map id transform time: " + (perf.faissIdMapDoneTime - perf.faissIdMapIdTransformDoneTime));

            //
            // Index HNSW
            //
            System.out.println("# Index HNSW flat");
            System.out.println("  Total execution time: " + (perf.indexHnswFlatDoneTime - perf.indexHnswFlatStartTime));
            System.out.println("  Before HNSW searching took: " + (perf.hnswSearchStartTime - perf.indexHnswFlatStartTime));
            System.out.println("  HNSW searching took: " + (perf.hnswSearchDoneTime - perf.hnswSearchStartTime));
            System.out.println("  Distance negation took: " + (perf.indexHnswFlatDoneTime - perf.hnswSearchDoneTime));

            //
            // FAISS HNSW
            //
            System.out.println("# FAISS HNSW start || entry_point=" + perf.hnswEntryPoint);
            System.out.println("  Total execution time: " + (perf.faissHnswDoneTime - perf.faissHnswStartTime));
            System.out.println("  Total vectors visit during greedy: " + perf.numVisitsDuringGreedy);
            System.out.println("  Total vectors visit: " + perf.numVisits);
            System.out.println("  Greedy took: " + (perf.hnswGreedyDoneTime - perf.faissHnswStartTime));
            System.out.println(
                "  After greedy, entry point=" + perf.entryPointAfterGreedy + ", dist=" + perf.entryPointAfterGreedyDistance);
            System.out.println("  BFS took: " + (perf.hnswBfsDoneTime - perf.hnswGreedyDoneTime));
            System.out.println("  Order results took: " + (perf.faissHnswDoneTime - perf.hnswBfsDoneTime));
        }
    }

    public static void setHnswEntryPoint(int entryPoint) {
        if (TURN_ON) {
            KdyPerfCheck perf = THREAD_LOCAL.get();
            perf.hnswEntryPoint = entryPoint;
        }
    }

    public static void incVectorVisit() {
        if (TURN_ON) {
            KdyPerfCheck perf = THREAD_LOCAL.get();
            perf.numVisits++;
        }
    }
}

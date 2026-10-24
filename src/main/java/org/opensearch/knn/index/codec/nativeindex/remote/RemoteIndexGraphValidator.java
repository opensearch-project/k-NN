/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import org.apache.lucene.index.MergeAbortChecker;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.index.codec.nativeindex.IndexBuildAbortedException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;

import java.io.EOFException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

/** Validates serialized FAISS graph structure and weak connectivity before accepting a remote build. */
final class RemoteIndexGraphValidator {
    private RemoteIndexGraphValidator() {}

    /** A detected graph fault, distinct from an I/O failure while reading the staged artifact. */
    static final class GraphFaultException extends IOException {
        GraphFaultException(String message) {
            super(message);
        }

        GraphFaultException(String message, Throwable cause) {
            super(message, cause);
        }
    }

    static void validate(
        IndexInput input,
        int expectedVectorCount,
        int expectedDimension,
        int expectedMaxDoc,
        String expectedVectorDataType,
        String expectedSpaceType,
        boolean skipStoredVectors
    ) throws IOException {
        try {
            checkCancellation();
            if (expectedVectorCount <= 0 || expectedMaxDoc <= 0) {
                throw new GraphFaultException("Remote graph and target segment must contain vectors");
            }
            GraphLayout graph = readLayout(
                input,
                expectedVectorCount,
                expectedDimension,
                expectedMaxDoc,
                expectedVectorDataType,
                expectedSpaceType,
                skipStoredVectors
            );
            validateConnectivity(input, graph, expectedVectorCount);
        } catch (GraphFaultException e) {
            throw e;
        } catch (IOException | RuntimeException e) {
            // EOF in a completely downloaded artifact is corruption. Preserve other I/O failures, including cancellation.
            IOException ioFailure = null;
            for (Throwable cause = e; cause != null; cause = cause.getCause()) {
                if (cause instanceof EOFException) {
                    throw new GraphFaultException("Remote graph is truncated", e);
                }
                if (cause instanceof IOException failure) {
                    ioFailure = failure;
                }
            }
            if (ioFailure != null) {
                throw ioFailure;
            }
            throw new GraphFaultException("Remote graph has invalid serialized data", e);
        }
    }

    private record Header(int dimension, int codeSize, int metric) {
        boolean matches(Header other) {
            // Binary graphs may retain the default Hamming metric while their ID map records the ADC metric.
            // Float headers, identified by the absence of a binary code size, must agree on the metric as well.
            return dimension == other.dimension && codeSize == other.codeSize && (codeSize > 0 || metric == other.metric);
        }
    }

    private record Section(long offset, long count) {
    }

    private record GraphLayout(Section cumulative, Section levels, Section offsets, Section neighbors, int maxLevel, int entryPoint,
        boolean baseLevelOnly) {
    }

    /**
     * Read the supported FAISS layouts without the search loader's metadata allocations or recursive index loading.
     * Sections are counted arrays: a little-endian 64-bit element count followed by fixed-width elements.
     */
    private static GraphLayout readLayout(
        IndexInput input,
        int expectedVectorCount,
        int expectedDimension,
        int expectedMaxDoc,
        String expectedVectorDataType,
        String expectedSpaceType,
        boolean skipStoredVectors
    ) throws IOException {
        String type = readType(input);
        if (type.equals("IxMp") == false && type.equals("IBMp") == false) {
            throw new GraphFaultException("Remote graph must have a document ID map");
        }
        boolean binary = type.equals("IBMp");
        if (binary != VectorDataType.BINARY.getValue().equals(expectedVectorDataType)) {
            throw new GraphFaultException("Remote graph encoding does not match the build request");
        }
        Header mapHeader = readHeader(input, binary, expectedVectorCount);
        if (mapHeader.dimension != expectedDimension) {
            throw new GraphFaultException("Remote graph dimension does not match the build request");
        }
        type = readType(input);
        binary = type.equals("IBHf") || type.equals("IBHc");
        boolean cagra = type.equals("IHNc") || type.equals("IHc2") || type.equals("IBHc");
        if (cagra == false && type.equals("IHNf") == false && type.equals("IHNs") == false && type.equals("IBHf") == false) {
            throw new GraphFaultException("Unsupported remote graph type: " + type);
        }
        Header header = readHeader(input, binary, expectedVectorCount);
        if (mapHeader.matches(header) == false) {
            throw new GraphFaultException("Remote graph header does not match its ID map");
        }
        if (binary == false && header.metric != expectedFloatMetric(expectedSpaceType)) {
            throw new GraphFaultException("Remote graph metric does not match the build request");
        }
        if (skipStoredVectors && binary && mapHeader.metric != 0 && mapHeader.metric != 1) {
            throw new GraphFaultException("Remote binary graph has an unsupported ADC metric: " + mapHeader.metric);
        }
        boolean baseLevelOnly = false;
        if (cagra) {
            requireBytes(input, type.equals("IHc2") ? 10 : 6);
            int keepMaxSizeLevel0 = input.readByte();
            int baseLevelOnlyFlag = input.readByte();
            if ((keepMaxSizeLevel0 != 0 && keepMaxSizeLevel0 != 1) || (baseLevelOnlyFlag != 0 && baseLevelOnlyFlag != 1)) {
                throw new GraphFaultException("Remote CAGRA graph has invalid search flags");
            }
            baseLevelOnly = baseLevelOnlyFlag == 1;
            int entryPoints = input.readInt();
            // Memory-optimized search always uses random CAGRA entry points, including full HNSW artifacts.
            if (entryPoints <= 0) {
                throw new GraphFaultException("Remote CAGRA graph has no search entry points");
            }
            if (type.equals("IHc2")) {
                input.readInt(); // Numeric type; storage below describes its representation.
            }
        }
        readSection(input, Double.BYTES); // Assignment probabilities, used only when building the graph.
        Section cumulative = readSection(input, Integer.BYTES);
        Section levels = readSection(input, Integer.BYTES);
        Section offsets = readSection(input, Long.BYTES);
        Section neighbors = readSection(input, Integer.BYTES);
        if (cumulative.count < 2 || levels.count != expectedVectorCount || offsets.count != (long) expectedVectorCount + 1) {
            throw new GraphFaultException("Remote graph has inconsistent serialized sections");
        }
        requireBytes(input, 5L * Integer.BYTES);
        int entryPoint = input.readInt();
        int maxLevel = input.readInt();
        input.readInt(); // efConstruction
        input.readInt(); // efSearch
        input.readInt(); // Deprecated search parameter.
        int minimumEntryPointAndLevel = baseLevelOnly ? -1 : 0;
        if (entryPoint < minimumEntryPointAndLevel
            || entryPoint >= expectedVectorCount
            || maxLevel < minimumEntryPointAndLevel
            || maxLevel >= cumulative.count - 1) {
            throw new GraphFaultException("Remote graph has invalid entry point or maximum level");
        }
        // FaissFlatIndexFactory can supply local storage for float CAGRA and ordinary binary HNSW only.
        boolean supportsSkippedStorage = (cagra && binary == false) || type.equals("IBHf");
        readStorage(input, header, expectedVectorCount, expectedVectorDataType, skipStoredVectors, supportsSkippedStorage);
        Section ids = readSection(input, Long.BYTES);
        if (ids.count != expectedVectorCount) {
            throw new GraphFaultException("Remote graph ID map does not match the expected vector count");
        }
        input.seek(ids.offset);
        long previous = -1;
        for (int i = 0; i < expectedVectorCount; i++) {
            checkCancellation(i);
            long id = input.readLong();
            if (id < 0 || id < previous || id >= expectedMaxDoc) {
                throw new GraphFaultException("Remote graph has an invalid document ID: " + id);
            }
            previous = id;
        }
        // Existing FAISS fixtures and persisted indices may include a Lucene footer after the raw FAISS payload.
        if (input.length() - input.getFilePointer() == CodecUtil.footerLength()) {
            try {
                CodecUtil.retrieveChecksum(input);
            } catch (CorruptIndexException e) {
                throw new GraphFaultException("Remote graph has an invalid footer", e);
            }
        } else if (input.getFilePointer() != input.length()) {
            throw new GraphFaultException("Remote graph has unexpected trailing data");
        }
        return new GraphLayout(cumulative, levels, offsets, neighbors, maxLevel, entryPoint, baseLevelOnly);
    }

    private static int expectedFloatMetric(String expectedSpaceType) throws GraphFaultException {
        final SpaceType spaceType;
        try {
            spaceType = SpaceType.getSpace(expectedSpaceType);
        } catch (IllegalArgumentException e) {
            throw new GraphFaultException("Remote build request has an unsupported space type: " + expectedSpaceType, e);
        }
        return switch (spaceType) {
            case L2 -> 1;
            case INNER_PRODUCT, COSINESIMIL -> 0;
            default -> throw new GraphFaultException("Remote build request has an unsupported float space type: " + expectedSpaceType);
        };
    }

    private static Header readHeader(IndexInput input, boolean binary, int expectedVectorCount) throws IOException {
        requireBytes(input, binary ? 21 : 33);
        int dimension = input.readInt();
        int codeSize = binary ? input.readInt() : 0;
        long vectorCount = input.readLong();
        if (dimension <= 0 || vectorCount != expectedVectorCount || (binary && codeSize != (dimension + 7L) / 8)) {
            throw new GraphFaultException("Remote graph header does not match the expected vector count or has invalid dimensions");
        }
        if (binary == false) {
            input.readLong();
            input.readLong(); // Deprecated fields.
        }
        input.readByte(); // is_trained
        int metric = input.readInt();
        if (binary == false && metric != 0 && metric != 1) {
            throw new GraphFaultException("Remote graph has an unsupported metric: " + metric);
        }
        return new Header(dimension, codeSize, metric);
    }

    private static void readStorage(
        IndexInput input,
        Header graphHeader,
        int expectedVectorCount,
        String expectedVectorDataType,
        boolean skipStoredVectors,
        boolean supportsSkippedStorage
    ) throws IOException {
        String type = readType(input);
        if (type.equals("null")) {
            if (skipStoredVectors == false || supportsSkippedStorage == false) {
                throw new GraphFaultException("Remote graph cannot use skipped vector storage for this build");
            }
            return;
        }
        if (skipStoredVectors) {
            throw new GraphFaultException("Remote graph contains vector storage despite requesting skipped storage");
        }
        boolean binary = type.equals("IBxF");
        if (binary == false && type.equals("IxF2") == false && type.equals("IxFI") == false && type.equals("IxSQ") == false) {
            throw new GraphFaultException("Unsupported remote flat storage type: " + type);
        }
        Header header = readHeader(input, binary, expectedVectorCount);
        if (header.matches(graphHeader) == false) {
            throw new GraphFaultException("Remote flat storage does not match its graph header");
        }
        if (type.equals("IxF2") || type.equals("IxFI")) {
            if (VectorDataType.FLOAT.getValue().equals(expectedVectorDataType) == false) {
                throw new GraphFaultException("Remote flat storage encoding does not match the build request");
            }
        }
        long elementsPerVector = binary ? header.codeSize : header.dimension;
        int elementSize = binary ? Byte.BYTES : Float.BYTES;
        if (type.equals("IxSQ")) {
            requireBytes(input, 3L * Integer.BYTES + 2L * Long.BYTES);
            int quantizer = input.readInt();
            int rangeStatistic = input.readInt();
            input.readInt(); // Floating-point range argument.
            long dimension = input.readLong();
            long codeSize = input.readLong();
            // The search loader supports FP16, BF16 and signed byte scalar quantizers.
            int bytesPerDimension = switch (quantizer) {
                case 4, 7 -> 2;
                case 8 -> 1;
                default -> throw new GraphFaultException("Unsupported remote scalar quantizer: " + quantizer);
            };
            boolean encodingMatches = switch (quantizer) {
                case 4 -> VectorDataType.HALF_FLOAT.getValue().equals(expectedVectorDataType);
                case 7 -> VectorDataType.FLOAT.getValue().equals(expectedVectorDataType);
                case 8 -> VectorDataType.BYTE.getValue().equals(expectedVectorDataType);
                default -> false;
            };
            if (encodingMatches == false) {
                throw new GraphFaultException("Remote scalar storage encoding does not match the build request");
            }
            if (rangeStatistic < 0 || rangeStatistic > 3 || dimension != header.dimension || codeSize != dimension * bytesPerDimension) {
                throw new GraphFaultException("Remote scalar storage has inconsistent metadata");
            }
            readSection(input, Float.BYTES); // Trained scalar quantizer values.
            elementsPerVector = codeSize;
            elementSize = Byte.BYTES;
        }
        Section vectors = readSection(input, elementSize);
        if (vectors.count != elementsPerVector * expectedVectorCount) {
            throw new GraphFaultException("Remote flat storage length does not match its vector count");
        }
    }

    private static Section readSection(IndexInput input, int elementSize) throws IOException {
        requireBytes(input, Long.BYTES);
        long count = input.readLong();
        long offset = input.getFilePointer();
        // Division before multiplication avoids overflow and prevents seeks outside the staged file.
        if (count < 0 || count > (input.length() - offset) / elementSize) {
            throw new GraphFaultException("Remote graph has an invalid or truncated section");
        }
        input.seek(offset + count * elementSize);
        return new Section(offset, count);
    }

    private static String readType(IndexInput input) throws IOException {
        requireBytes(input, 4);
        byte[] type = new byte[4];
        input.readBytes(type, 0, type.length);
        return new String(type, StandardCharsets.US_ASCII);
    }

    private static void requireBytes(IndexInput input, long count) throws IOException {
        if (input.getFilePointer() < 0 || count > input.length() - input.getFilePointer()) {
            throw new GraphFaultException("Remote graph is truncated");
        }
    }

    private static void validateConnectivity(IndexInput input, GraphLayout graph, int count) throws IOException {
        // Scan metadata and adjacency lists directly. Heap use is one int per expected vector, independent of file-supplied sizes.
        try (
            IndexInput offsets = input.clone();
            IndexInput levels = input.clone();
            IndexInput cumulative = input.clone();
            IndexInput neighborLevels = input.clone()
        ) {
            // Keep cancellation polling proportional to total validation work. Per-loop counters restart for every node
            // and level, so small adjacency lists would otherwise never reach the polling interval.
            long work = 0;
            cumulative.seek(graph.cumulative.offset);
            if (cumulative.readInt() != 0) {
                throw new GraphFaultException("Remote graph cumulative neighbor table must start at zero");
            }
            int previousDegree = 0;
            for (long i = 1; i < graph.cumulative.count; i++) {
                checkCancellation(work++);
                int degree = cumulative.readInt();
                if (degree <= previousDegree) {
                    throw new GraphFaultException("Remote graph cumulative neighbor table is invalid");
                }
                previousDegree = degree;
            }
            checkCancellation();
            int[] parents = new int[count];
            Arrays.fill(parents, -1);
            int components = count;
            offsets.seek(graph.offsets.offset);
            levels.seek(graph.levels.offset);
            long start = offsets.readLong();
            if (start != 0) {
                throw new GraphFaultException("Remote graph neighbor offsets must start at zero");
            }
            int highestLevel = 0;
            for (int node = 0; node < count; node++) {
                checkCancellation(work++);
                long end = offsets.readLong();
                int nodeLevels = levels.readInt();
                if (end <= start || end > graph.neighbors.count || nodeLevels < 1 || nodeLevels >= graph.cumulative.count) {
                    throw new GraphFaultException("Remote graph has invalid neighbor offsets or levels");
                }
                if (graph.baseLevelOnly) {
                    // Native CAGRA range search still traverses maxLevel from the serialized entry point, or a random
                    // vertex when entryPoint=-1. Every possible starting vertex must therefore have that level.
                    if ((graph.entryPoint == -1 || node == graph.entryPoint) && nodeLevels <= graph.maxLevel) {
                        throw new GraphFaultException("Remote CAGRA range search entry point does not exist at the maximum level");
                    }
                } else if (node == graph.entryPoint && nodeLevels - 1 != graph.maxLevel) {
                    throw new GraphFaultException("Remote graph entry point does not exist at the maximum level");
                }
                highestLevel = Math.max(highestLevel, nodeLevels - 1);
                cumulative.seek(graph.cumulative.offset + (long) nodeLevels * Integer.BYTES);
                if (end - start != cumulative.readInt()) {
                    throw new GraphFaultException("Remote graph neighbor offsets do not match its levels");
                }
                cumulative.seek(graph.cumulative.offset);
                int levelStart = cumulative.readInt();
                for (int level = 0; level < nodeLevels; level++) {
                    checkCancellation(work++);
                    int levelEnd = cumulative.readInt();
                    input.seek(graph.neighbors.offset + (start + levelStart) * Integer.BYTES);
                    for (int i = levelStart; i < levelEnd; i++) {
                        checkCancellation(work++);
                        int neighbor = input.readInt();
                        if (neighbor == -1) {
                            break;
                        }
                        if (neighbor < 0 || neighbor >= count) {
                            throw new GraphFaultException("Remote graph contains an invalid neighbor ID: " + neighbor);
                        }
                        if (level > 0) {
                            neighborLevels.seek(graph.levels.offset + (long) neighbor * Integer.BYTES);
                            if (neighborLevels.readInt() <= level) {
                                throw new GraphFaultException("Remote graph neighbor does not exist on level " + level);
                            }
                        }
                        if (level == 0) {
                            int nodeRoot = root(parents, node);
                            int neighborRoot = root(parents, neighbor);
                            if (nodeRoot != neighborRoot) {
                                if (parents[nodeRoot] > parents[neighborRoot]) {
                                    int temp = nodeRoot;
                                    nodeRoot = neighborRoot;
                                    neighborRoot = temp;
                                }
                                parents[nodeRoot] += parents[neighborRoot];
                                parents[neighborRoot] = nodeRoot;
                                components--;
                            }
                        }
                    }
                    levelStart = levelEnd;
                }
                start = end;
            }
            if (start != graph.neighbors.count || (graph.baseLevelOnly == false && highestLevel != graph.maxLevel)) {
                throw new GraphFaultException("Remote graph has inconsistent terminal offset or maximum level");
            }
            if (components > 1) {
                throw new GraphFaultException("Remote graph has " + components + " connected components; falling back to CPU indexing");
            }
        }
        checkCancellation();
    }

    private static void checkCancellation(long iteration) throws IndexBuildAbortedException {
        if ((iteration & 1023) == 0) {
            checkCancellation();
        }
    }

    private static void checkCancellation() throws IndexBuildAbortedException {
        if (MergeAbortChecker.isMergeAborted()) {
            throw new IndexBuildAbortedException("Remote graph validation aborted due to merge cancellation");
        }
    }

    private static int root(int[] parents, int node) {
        while (parents[node] >= 0) {
            int parent = parents[node];
            if (parents[parent] >= 0) {
                parents[node] = parents[parent];
            }
            node = parent;
        }
        return node;
    }
}

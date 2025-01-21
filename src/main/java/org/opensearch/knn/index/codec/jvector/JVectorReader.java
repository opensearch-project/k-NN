/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.*;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.*;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80CompoundDirectory;

import java.io.IOException;
import java.nio.file.Path;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.readVectorEncoding;

@Log4j2
public class JVectorReader extends KnnVectorsReader {
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();

    private final FieldInfos fieldInfos;
    private final String indexDataFileName;
    private final String baseDataFileName;
    private final Path directoryBasePath;
    // Maps field name to field entries
    private final Map<String, FieldEntry> fieldEntryMap = new HashMap<>(1);
    private final Directory directory;
    private final SegmentReadState state;


    public JVectorReader(SegmentReadState state) throws IOException {
        this.state = state;
        this.fieldInfos = state.fieldInfos;
        this.baseDataFileName = state.segmentInfo.name + "_" + state.segmentSuffix;
        String metaFileName =
                IndexFileNames.segmentFileName(
                        state.segmentInfo.name, state.segmentSuffix, JVectorFormat.META_EXTENSION);
        this.directory = state.directory;
        this.directoryBasePath = resolveDirectoryPath(directory);
        boolean success = false;
        try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaFileName, state.context)) {
            CodecUtil.checkIndexHeader(
                    meta,
                    JVectorFormat.META_CODEC_NAME,
                    JVectorFormat.VERSION_START,
                    JVectorFormat.VERSION_CURRENT,
                    state.segmentInfo.getId(),
                    state.segmentSuffix);
            Set<String> filenames = state.segmentInfo.files();
            readFields(meta);
            CodecUtil.checkFooter(meta);

            this.indexDataFileName =
                    IndexFileNames.segmentFileName(
                            state.segmentInfo.name,
                            state.segmentSuffix,
                            JVectorFormat.VECTOR_INDEX_EXTENSION);

            success = true;
        } finally {
            if (!success) {
                IOUtils.closeWhileHandlingException(this);
            }
        }
    }

    @Override
    public void checkIntegrity() throws IOException {
        // This is already done when loading the fields
        // TODO: Implement this, for now this will always pass
        //CodecUtil.checksumEntireFile(vectorIndex);
    }

    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
        return new JVectorFloatVectorValues(fieldEntryMap.get(field).index, fieldEntryMap.get(field).similarityFunction);
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
        /**
         * Byte vector values are not supported in jVector library. Instead use PQ.
         */
        return null;
    }

    @Override
    public void search(String field, float[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {
        final OnDiskGraphIndex index = fieldEntryMap.get(field).index;

        // search for a random vector using a GraphSearcher and SearchScoreProvider
        VectorFloat<?> q = VECTOR_TYPE_SUPPORT.createFloatVector(target);
        try (GraphSearcher searcher = new GraphSearcher(index)) {
            SearchScoreProvider ssp = SearchScoreProvider.exact(q, fieldEntryMap.get(field).similarityFunction, index.getView());
            SearchResult sr = searcher.search(ssp, knnCollector.k(), io.github.jbellis.jvector.util.Bits.ALL);
            for (SearchResult.NodeScore ns : sr.getNodes()) {
                knnCollector.collect(ns.node, ns.score);
            }
        }
    }

    @Override
    public void search(String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {
        // TODO: implement this
    }

    @Override
    public void close() throws IOException {
        for (FieldEntry fieldEntry : fieldEntryMap.values()) {
            IOUtils.close(fieldEntry.readerSupplier::close);
        }
    }

    @Override
    public long ramBytesUsed() {
        return 0;
    }


    private void readFields(ChecksumIndexInput meta) throws IOException {
        for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
            final FieldInfo fieldInfo = fieldInfos.fieldInfo(fieldNumber); // read field number)
            final VectorEncoding vectorEncoding = readVectorEncoding(meta);
            final VectorSimilarityFunction similarityFunction = VectorSimilarityMapper.ordToDistFunc(meta.readInt());
            final long vectorIndexOffset = meta.readVLong();
            final long vectorIndexLength = meta.readVLong();
            final int dimension = meta.readVInt();
            fieldEntryMap.put(fieldInfo.name, new FieldEntry(fieldInfo, similarityFunction, vectorEncoding, vectorIndexOffset, vectorIndexLength, dimension));
        }
    }

    class FieldEntry {
        private final FieldInfo fieldInfo;
        private final VectorEncoding vectorEncoding;
        private final VectorSimilarityFunction similarityFunction;
        private final long vectorIndexOffset;
        private final long vectorIndexLength;
        private final int dimension;
        private final ReaderSupplier readerSupplier;
        private final OnDiskGraphIndex index;

        public FieldEntry(
                FieldInfo fieldInfo,
                VectorSimilarityFunction similarityFunction,
                VectorEncoding vectorEncoding,
                long vectorIndexOffset,
                long vectorIndexLength,
                int dimension) throws IOException {
            this.fieldInfo = fieldInfo;
            this.similarityFunction = similarityFunction;
            this.vectorEncoding = vectorEncoding;
            this.vectorIndexOffset = vectorIndexOffset;
            this.vectorIndexLength = vectorIndexLength;
            this.dimension = dimension;
            // TODO: do not depend on the actual nio.Path switch to file name only!
            final Path expectedIndexFilePath = JVectorFormat.getVectorIndexPath(directoryBasePath, baseDataFileName, fieldInfo.name);
            final String originalIndexFileName = expectedIndexFilePath.getFileName().toString();
            final Path indexFilePath;
            final long sliceOffset;
            if (state.segmentInfo.getUseCompoundFile()) {
                if (directory instanceof JVectorCompoundFormat.JVectorCompoundReader) {
                    JVectorCompoundFormat.JVectorCompoundReader jVectorCompoundReader = (JVectorCompoundFormat.JVectorCompoundReader) directory;
                    sliceOffset = jVectorCompoundReader.getOffsetInCompoundFile(originalIndexFileName);
                    indexFilePath = jVectorCompoundReader.getCompoundFilePath();
                } else if (directory instanceof KNN80CompoundDirectory) {
                    KNN80CompoundDirectory knn80CompoundDirectory = (KNN80CompoundDirectory) directory;
                    JVectorCompoundFormat.JVectorCompoundReader jVectorCompoundReader = new JVectorCompoundFormat.JVectorCompoundReader(
                            knn80CompoundDirectory.getDelegate(),
                            knn80CompoundDirectory.getDir(),
                            state.segmentInfo,
                            state.context
                    );
                    sliceOffset = jVectorCompoundReader.getOffsetInCompoundFile(originalIndexFileName);
                    indexFilePath = jVectorCompoundReader.getCompoundFilePath();
                } else {
                    throw new IllegalArgumentException("directory must be JVectorCompoundFormat or KNN80CompoundDirectory but instead had type: " + directory.getClass().getName());
                }

            } else {
                sliceOffset = 0;
                indexFilePath = expectedIndexFilePath;
            }

            // Check the header
            try (IndexInput indexInput = directory.openInput(originalIndexFileName, state.context)) {
                CodecUtil.checkIndexHeader(
                        indexInput,
                        JVectorFormat.VECTOR_INDEX_CODEC_NAME,
                        JVectorFormat.VERSION_START,
                        JVectorFormat.VERSION_CURRENT,
                        state.segmentInfo.getId(),
                        state.segmentSuffix);
            }

            // Load the graph index
            this.readerSupplier = ReaderSupplierFactory.open(indexFilePath);
            this.index = OnDiskGraphIndex.load(readerSupplier, sliceOffset + vectorIndexOffset);

            // Check the footer
            try (ChecksumIndexInput indexInput = directory.openChecksumInput(originalIndexFileName, state.context)) {
                indexInput.seek(vectorIndexOffset + vectorIndexLength);
                CodecUtil.checkFooter(indexInput);
            }

        }
    }


    /**
     * Utility class to map between Lucene and jVector similarity functions and metadata ordinals.
     */
    public static class VectorSimilarityMapper {
        /**
         List of vector similarity functions supported by <a href="https://github.com/jbellis/jvector">jVector library</a>
         The similarity functions orders matter in this list because it is later used to resolve the similarity function by ordinal.
         */
        public static final List<VectorSimilarityFunction> JVECTOR_SUPPORTED_SIMILARITY_FUNCTIONS =
                List.of(
                        VectorSimilarityFunction.EUCLIDEAN,
                        VectorSimilarityFunction.DOT_PRODUCT,
                        VectorSimilarityFunction.COSINE);

        public static final Map<org.apache.lucene.index.VectorSimilarityFunction, VectorSimilarityFunction> luceneToJVectorMap = Map.of(
                org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN, VectorSimilarityFunction.EUCLIDEAN,
                org.apache.lucene.index.VectorSimilarityFunction.DOT_PRODUCT, VectorSimilarityFunction.DOT_PRODUCT,
                org.apache.lucene.index.VectorSimilarityFunction.COSINE, VectorSimilarityFunction.COSINE
        );

        public static int distFuncToOrd(org.apache.lucene.index.VectorSimilarityFunction func) {
            if (luceneToJVectorMap.containsKey(func)) {
                return JVECTOR_SUPPORTED_SIMILARITY_FUNCTIONS.indexOf(luceneToJVectorMap.get(func));
            }

            throw new IllegalArgumentException("invalid distance function: " + func);
        }

        public static VectorSimilarityFunction ordToDistFunc(int ord) {
            return JVECTOR_SUPPORTED_SIMILARITY_FUNCTIONS.get(ord);
        }
    }

    public static Path resolveDirectoryPath(Directory dir) {
        while (!(dir instanceof FSDirectory)) {
            final String dirType = dir.getClass().getName();
            log.info("unwrapping dir of type: {} to find path", dirType);
            if (dir instanceof FilterDirectory) {
                dir = ((FilterDirectory) dir).getDelegate();
            } else if (dir instanceof JVectorCompoundFormat.JVectorCompoundReader) {
                return ((JVectorCompoundFormat.JVectorCompoundReader) dir).getDirectoryBasePath();
            } else if (dir instanceof KNN80CompoundDirectory) {
                dir = ((KNN80CompoundDirectory) dir).getDir();
            } else {
                throw new IllegalArgumentException("directory must be FSDirectory or a wrapper around it but instead had type: " + dirType);
            }
        }
        final Path path = ((FSDirectory) dir).getDirectory();
        log.info("resolved directory path from FSDirectory: {}", path);
        return path;
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.*;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.*;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.*;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.common.collect.Tuple;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

@Log4j2
public class JVectorWriter extends KnnVectorsWriter {
    private static final long SHALLOW_RAM_BYTES_USED =
            RamUsageEstimator.shallowSizeOfInstance(JVectorWriter.class);
    private final List<JVectorWriter.FieldWriter<?>> fields = new ArrayList<>();

    private final IndexOutput meta;
    private final IndexOutput vectorIndex;
    private final String indexDataFileName;
    private final String baseDataFileName;
    private final Path directoryBasePath;
    private final SegmentWriteState segmentWriteState;
    private final int maxConn;
    private final int beamWidth;
    private final float degreeOverflow;
    private final float alpha;
    private boolean finished = false;


    public JVectorWriter(SegmentWriteState segmentWriteState, int maxConn, int beamWidth, float degreeOverflow, float alpha) throws IOException {
        this.segmentWriteState = segmentWriteState;
        this.maxConn = maxConn;
        this.beamWidth = beamWidth;
        this.degreeOverflow = degreeOverflow;
        this.alpha = alpha;
        String metaFileName =
                IndexFileNames.segmentFileName(
                        segmentWriteState.segmentInfo.name, segmentWriteState.segmentSuffix, JVectorFormat.META_EXTENSION);

        this.indexDataFileName =
                IndexFileNames.segmentFileName(
                        segmentWriteState.segmentInfo.name,
                        segmentWriteState.segmentSuffix,
                        JVectorFormat.VECTOR_INDEX_EXTENSION);
        this.baseDataFileName = segmentWriteState.segmentInfo.name + "_" + segmentWriteState.segmentSuffix;

        Directory dir = segmentWriteState.directory;
        this.directoryBasePath = JVectorReader.resolveDirectoryPath(dir);

        boolean success = false;
        try {
            meta = segmentWriteState.directory.createOutput(metaFileName, segmentWriteState.context);
            vectorIndex = segmentWriteState.directory.createOutput(indexDataFileName, segmentWriteState.context);
            CodecUtil.writeIndexHeader(
                    meta,
                    JVectorFormat.META_CODEC_NAME,
                    JVectorFormat.VERSION_CURRENT,
                    segmentWriteState.segmentInfo.getId(),
                    segmentWriteState.segmentSuffix);

            CodecUtil.writeIndexHeader(
                    vectorIndex,
                    JVectorFormat.VECTOR_INDEX_CODEC_NAME,
                    JVectorFormat.VERSION_CURRENT,
                    segmentWriteState.segmentInfo.getId(),
                    segmentWriteState.segmentSuffix);

            success = true;
        } finally {
            if (!success) {
                IOUtils.closeWhileHandlingException(this);
            }
        }
    }

    @Override
    public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
        log.info("Adding field {} in segment {}", fieldInfo.name, segmentWriteState.segmentInfo.name);
        if (fieldInfo.getVectorEncoding() == VectorEncoding.BYTE) {
            final String errorMessage = "byte[] vectors are not supported in JVector. " +
                    "Instead you should only use float vectors and leverage product quantization during indexing." +
                    "This can provides much greater savings in storage and memory";
            log.error(errorMessage);
            throw new UnsupportedOperationException(errorMessage);
        }
        JVectorWriter.FieldWriter<?> newField =
                new JVectorWriter.FieldWriter<>(fieldInfo, segmentWriteState.segmentInfo.name);
        fields.add(newField);
        return newField;
    }

    @Override
    public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        log.info("Merging field {} into segment {}", fieldInfo.name, segmentWriteState.segmentInfo.name);
        var success = false;
        try {
            switch (fieldInfo.getVectorEncoding()) {
                case BYTE:
                    var byteWriter =
                            (JVectorWriter.FieldWriter<byte[]>) addField(fieldInfo);
                    ByteVectorValues mergedBytes =
                            MergedVectorValues.mergeByteVectorValues(fieldInfo, mergeState);
                    for (int doc = mergedBytes.nextDoc();
                         doc != DocIdSetIterator.NO_MORE_DOCS;
                         doc = mergedBytes.nextDoc()) {
                        byteWriter.addValue(doc, mergedBytes.vectorValue());
                    }
                    writeField(byteWriter);
                    break;
                case FLOAT32:
                    var floatVectorFieldWriter =
                            (JVectorWriter.FieldWriter<float[]>) addField(fieldInfo);
                    int baseDocId = 0;
                    for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
                        FloatVectorValues floatVectorValues = mergeState.knnVectorsReaders[i].getFloatVectorValues(fieldInfo.name);
                        var floatVectors = new ArrayList<float[]>();
                        for (int doc = floatVectorValues.nextDoc();
                             doc != DocIdSetIterator.NO_MORE_DOCS;
                             doc = floatVectorValues.nextDoc()) {
                            floatVectors.add(floatVectorValues.vectorValue());
                        }
                        for (int doc = 0; doc < floatVectors.size(); doc++) {
                            floatVectorFieldWriter.addValue(baseDocId + doc, floatVectors.get(doc));
                        }

                        baseDocId += floatVectorValues.size();
                    }
                    writeField(floatVectorFieldWriter);
                    break;
            }
            success = true;
            log.info("Completed Merge field {} into segment {}", fieldInfo.name, segmentWriteState.segmentInfo.name);
        } finally {
            if (success) {
                //IOUtils.close(scorerSupplier);
            } else {
                //IOUtils.closeWhileHandlingException(scorerSupplier);
            }
        }
    }

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        log.info("Flushing {} fields", fields.size());

        for (JVectorWriter.FieldWriter<?> field : fields) {
            if (sortMap == null) {
                writeField(field);
            } else {
                throw new UnsupportedOperationException("Not implemented yet");
                //writeSortingField(field, sortMap);
            }
        }
    }

    private void writeField(JVectorWriter.FieldWriter<?> fieldData) throws IOException {
        // write graph
        //long vectorIndexOffset = vectorIndex.getFilePointer();
        OnHeapGraphIndex graph = fieldData.getGraph();
        Tuple<Long, Long>  vectorIndexOffsetAndLength = writeGraph(graph, fieldData);


        writeMeta(
                fieldData.fieldInfo,
                vectorIndexOffsetAndLength.v1(), // vectorIndexOffset
                vectorIndexOffsetAndLength.v2() // vectorIndexLength);
        );
    }



    private void writeMeta(
            FieldInfo field,
            long vectorIndexOffset,
            long vectorIndexLength)
            throws IOException {
        meta.writeInt(field.number);
        meta.writeInt(field.getVectorEncoding().ordinal());
        meta.writeInt(JVectorReader.VectorSimilarityMapper.distFuncToOrd(field.getVectorSimilarityFunction()));
        meta.writeVLong(vectorIndexOffset);
        meta.writeVLong(vectorIndexLength);
        meta.writeVInt(field.getVectorDimension());
    }



    /**
     * Writes the graph to the vector index file
     * @param graph graph
     * @param fieldData fieldData
     * @return Tuple of start offset and length of the graph
     * @throws IOException IOException
     */
    private Tuple<Long, Long> writeGraph(OnHeapGraphIndex graph, FieldWriter<?> fieldData) throws IOException {
        // TODO: use the vector index inputStream instead of this!
        final Path jvecFilePath = JVectorFormat.getVectorIndexPath(directoryBasePath, baseDataFileName, fieldData.fieldInfo.name);
        /** This is an ugly hack to make sure Lucene actually knows about our input stream files, otherwise it will delete them */
        IndexOutput indexOutput = segmentWriteState.directory.createOutput(jvecFilePath.getFileName().toString(), segmentWriteState.context);
        CodecUtil.writeIndexHeader(
                indexOutput,
                JVectorFormat.VECTOR_INDEX_CODEC_NAME,
                JVectorFormat.VERSION_CURRENT,
                segmentWriteState.segmentInfo.getId(),
                segmentWriteState.segmentSuffix);
        final long startOffset = indexOutput.getFilePointer();
        indexOutput.close();
        /** End of ugly hack */

        log.info("Writing graph to {}", jvecFilePath);
        final Tuple<Long, Long> result;
        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, jvecFilePath)
                .with(new InlineVectors(fieldData.randomAccessVectorValues.dimension()))
                .withStartOffset(startOffset)
                .build()) {
            var suppliers = Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
                    nodeId -> new InlineVectors.State(fieldData.randomAccessVectorValues.getVector(nodeId)));
            writer.write(suppliers);
            long endOffset = writer.getOutput().position();
            result = new Tuple<>(startOffset, endOffset - startOffset);
            // write footer by wrapping jVector RandomAccessOutput to IndexOutput object
            // This mostly helps to interface with the existing Lucene CodecUtil
            IndexOutput jvecIndexOutput = new JVectorIndexOutput(writer.getOutput());
            CodecUtil.writeFooter(jvecIndexOutput);
        }


        return result;
    }

    @Override
    public void finish() throws IOException {
        log.info("Finishing segment {}", segmentWriteState.segmentInfo.name);
        if (finished) {
            throw new IllegalStateException("already finished");
        }
        finished = true;

        if (meta != null) {
            // write end of fields marker
            meta.writeInt(-1);
            CodecUtil.writeFooter(meta);
        }

        if (vectorIndex != null) {
            CodecUtil.writeFooter(vectorIndex);
        }
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(meta, vectorIndex);
    }

    @Override
    public long ramBytesUsed() {
        long total = SHALLOW_RAM_BYTES_USED;
        for (JVectorWriter.FieldWriter<?> field : fields) {
            // the field tracks the delegate field usage
            total += field.ramBytesUsed();
        }
        return total;
    }

    class FieldWriter<T> extends KnnFieldVectorsWriter<T> {
        private final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();
        private final long SHALLOW_SIZE =
                RamUsageEstimator.shallowSizeOfInstance(JVectorWriter.FieldWriter.class);
        @Getter
        private final FieldInfo fieldInfo;
        private int lastDocID = -1;
        private final GraphIndexBuilder graphIndexBuilder;
        private final List<VectorFloat<?>> floatVectors = new ArrayList<>();
        private final String segmentName;
        private final RandomAccessVectorValues randomAccessVectorValues;
        private final BuildScoreProvider buildScoreProvider;


        FieldWriter(FieldInfo fieldInfo, String segmentName) {
            this.fieldInfo = fieldInfo;
            this.segmentName = segmentName;
            var originalDimension = fieldInfo.getVectorDimension();
            this.randomAccessVectorValues = new ListRandomAccessVectorValues(floatVectors, originalDimension);
            this.buildScoreProvider = BuildScoreProvider.randomAccessScoreProvider(randomAccessVectorValues, getVectorSimilarityFunction(fieldInfo));
            this.graphIndexBuilder = new GraphIndexBuilder(buildScoreProvider,
                    randomAccessVectorValues.dimension(),
                    maxConn,
                    beamWidth,
                    degreeOverflow,
                    alpha);
        }

        @Override
        public void addValue(int docID, T vectorValue) throws IOException {
            log.debug("Adding value {} to field {} in segment {}", vectorValue, fieldInfo.name, segmentName);
            if (docID == lastDocID) {
                throw new IllegalArgumentException(
                        "VectorValuesField \""
                                + fieldInfo.name
                                + "\" appears more than once in this document (only one value is allowed per field)");
            }
            if (vectorValue instanceof float[]) {
                var floats = (float[]) vectorValue;
                var vector = VECTOR_TYPE_SUPPORT.createFloatVector(floats);
                floatVectors.add(vector);
                graphIndexBuilder.addGraphNode(docID, vector);
            } else if (vectorValue instanceof byte[]) {
                final String errorMessage = "byte[] vectors are not supported in JVector. " +
                        "Instead you should only use float vectors and leverage product quantization during indexing." +
                        "This can provides much greater savings in storage and memory";
                log.error(errorMessage);
                throw new UnsupportedOperationException(errorMessage);
            } else {
                throw new IllegalArgumentException("Unsupported vector type: " + vectorValue.getClass());
            }


            lastDocID = docID;
        }

        @Override
        public T copyValue(T vectorValue) {
            throw new UnsupportedOperationException("copyValue not supported");
        }

        @Override
        public long ramBytesUsed() {
            return SHALLOW_SIZE
                    + graphIndexBuilder.getGraph().ramBytesUsed();
        }

        io.github.jbellis.jvector.vector.VectorSimilarityFunction getVectorSimilarityFunction(FieldInfo fieldInfo) {
            log.info("Matching vector similarity function {} for field {}", fieldInfo.getVectorSimilarityFunction(), fieldInfo.name);
            switch (fieldInfo.getVectorSimilarityFunction()) {
                case EUCLIDEAN:
                    return io.github.jbellis.jvector.vector.VectorSimilarityFunction.EUCLIDEAN;
                case COSINE:
                    return io.github.jbellis.jvector.vector.VectorSimilarityFunction.COSINE;
                case DOT_PRODUCT:
                    return io.github.jbellis.jvector.vector.VectorSimilarityFunction.DOT_PRODUCT;
                default:
                    throw new IllegalArgumentException("Unsupported similarity function: " + fieldInfo.getVectorSimilarityFunction());
            }
        }

        /**
         * This method will return the graph index for the field
         * @return OnHeapGraphIndex
         * @throws IOException IOException
         */
        public OnHeapGraphIndex getGraph() throws IOException {
            return graphIndexBuilder.getGraph();
        }
    }
}

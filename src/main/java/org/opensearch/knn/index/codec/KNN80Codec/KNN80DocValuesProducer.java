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

package org.opensearch.knn.index.codec.KNN80Codec;

import lombok.NonNull;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.opensearch.common.io.PathUtils;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapper.KNN_FIELD;

@Log4j2
public class KNN80DocValuesProducer extends DocValuesProducer {

    private final SegmentReadState state;
    private final DocValuesProducer delegate;
    private final NativeMemoryCacheManager nativeMemoryCacheManager;
    private final Map<String, String> indexPathMap = new HashMap();

    public KNN80DocValuesProducer(DocValuesProducer delegate, SegmentReadState state) {
        this.delegate = delegate;
        this.state = state;
        this.nativeMemoryCacheManager = NativeMemoryCacheManager.getInstance();

        Directory directory = state.directory;
        // directory would be CompoundDirectory, we need get directory firstly and then unwrap
        if (state.directory instanceof KNN80CompoundDirectory) {
            directory = ((KNN80CompoundDirectory) state.directory).getDir();
        }

        Directory dir = FilterDirectory.unwrap(directory);
        if (!(dir instanceof FSDirectory)) {
            log.warn("{} can not casting to FSDirectory", directory);
            return;
        }
        String directoryPath = ((FSDirectory) dir).getDirectory().toString();
        for (FieldInfo field : state.fieldInfos) {
            if (!field.attributes().containsKey(KNN_FIELD)) {
                continue;
            }
            // Only segments that contains BinaryDocValues and doesn't have vector values should be considered.
            // By default, we don't create BinaryDocValues for knn field anymore. However, users can set doc_values = true
            // to create binary doc values explicitly like any other field. Hence, we only want to include fields
            // where approximate search is possible only by BinaryDocValues.
            if (field.getDocValuesType() != DocValuesType.BINARY || field.hasVectorValues() == true) {
                continue;
            }
            // Only Native Engine put into indexPathMap
            KNNEngine knnEngine = getNativeKNNEngine(field);
            if (knnEngine == null) {
                continue;
            }
            List<String> engineFiles = KNNCodecUtil.getEngineFiles(knnEngine.getExtension(), field.name, state.segmentInfo);
            Path indexPath = PathUtils.get(directoryPath, engineFiles.get(0));
            indexPathMap.putIfAbsent(field.getName(), indexPath.toString());

        }
    }

    @Override
    public BinaryDocValues getBinary(FieldInfo field) throws IOException {
        return delegate.getBinary(field);
    }

    @Override
    public NumericDocValues getNumeric(FieldInfo field) throws IOException {
        return delegate.getNumeric(field);
    }

    @Override
    public SortedDocValues getSorted(FieldInfo field) throws IOException {
        return delegate.getSorted(field);
    }

    @Override
    public SortedNumericDocValues getSortedNumeric(FieldInfo field) throws IOException {
        return delegate.getSortedNumeric(field);
    }

    @Override
    public SortedSetDocValues getSortedSet(FieldInfo field) throws IOException {
        return delegate.getSortedSet(field);
    }

    @Override
    public void checkIntegrity() throws IOException {
        delegate.checkIntegrity();
    }

    @Override
    public void close() throws IOException {
        for (String path : indexPathMap.values()) {
            nativeMemoryCacheManager.invalidate(path);
        }
        delegate.close();
    }

    public final List<String> getOpenedIndexPath() {
        return new ArrayList<>(indexPathMap.values());
    }

    /**
     * Get KNNEngine From FieldInfo
     *
     * @param field which field we need produce from engine
     * @return if and only if Native Engine we return specific engine, else return null
     */
    private KNNEngine getNativeKNNEngine(@NonNull FieldInfo field) {

        final String modelId = field.attributes().get(MODEL_ID);
        if (modelId != null) {
            return null;
        }
        KNNEngine engine = FieldInfoExtractor.extractKNNEngine(field);
        if (KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(engine)) {
            return engine;
        }
        return null;
    }
}

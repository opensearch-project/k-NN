/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.SneakyThrows;
import org.apache.lucene.document.LateInteractionField;
import org.apache.lucene.index.IndexableField;
import org.opensearch.Version;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.settings.IndexScopedSettings;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.ContentPath;
import org.opensearch.index.mapper.Mapper;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;

import java.util.HashSet;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.Version.CURRENT;

public class LateInteractionFieldMapperTests extends KNNTestCase {

    private static final String FIELD_NAME = "tokens";
    private static final int DIMENSION = 4;

    @SneakyThrows
    public void testTypeParser_validMapping_thenSucceeds() {
        XContentBuilder mapping = XContentFactory.jsonBuilder()
            .startObject()
            .field(KNNConstants.TYPE, LateInteractionFieldMapper.CONTENT_TYPE)
            .field(KNNConstants.DIMENSION, DIMENSION)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.INNER_PRODUCT.getValue())
            .endObject();

        LateInteractionFieldMapper.Builder builder = (LateInteractionFieldMapper.Builder) LateInteractionFieldMapper.PARSER.parse(
            FIELD_NAME,
            xContentBuilderToMap(mapping),
            buildParserContext()
        );
        LateInteractionFieldMapper mapper = builder.build(new Mapper.BuilderContext(Settings.EMPTY, new ContentPath()));

        LateInteractionFieldType fieldType = mapper.fieldType();
        assertEquals(FIELD_NAME, fieldType.name());
        assertEquals(DIMENSION, fieldType.getDimension());
        assertEquals(SpaceType.INNER_PRODUCT, fieldType.getSpaceType());
        assertEquals(LateInteractionFieldMapper.CONTENT_TYPE, fieldType.typeName());
        assertTrue(fieldType.hasDocValues());
        assertFalse(fieldType.isSearchable());
    }

    @SneakyThrows
    public void testTypeParser_defaultSpaceType_thenInnerProduct() {
        XContentBuilder mapping = XContentFactory.jsonBuilder()
            .startObject()
            .field(KNNConstants.TYPE, LateInteractionFieldMapper.CONTENT_TYPE)
            .field(KNNConstants.DIMENSION, DIMENSION)
            .endObject();

        LateInteractionFieldMapper.Builder builder = (LateInteractionFieldMapper.Builder) LateInteractionFieldMapper.PARSER.parse(
            FIELD_NAME,
            xContentBuilderToMap(mapping),
            buildParserContext()
        );
        LateInteractionFieldMapper mapper = builder.build(new Mapper.BuilderContext(Settings.EMPTY, new ContentPath()));

        assertEquals(SpaceType.INNER_PRODUCT, mapper.fieldType().getSpaceType());
    }

    @SneakyThrows
    public void testTypeParser_missingDimension_thenThrows() {
        XContentBuilder mapping = XContentFactory.jsonBuilder()
            .startObject()
            .field(KNNConstants.TYPE, LateInteractionFieldMapper.CONTENT_TYPE)
            .endObject();

        LateInteractionFieldMapper.Builder builder = (LateInteractionFieldMapper.Builder) LateInteractionFieldMapper.PARSER.parse(
            FIELD_NAME,
            xContentBuilderToMap(mapping),
            buildParserContext()
        );
        expectThrows(IllegalArgumentException.class, () -> builder.build(new Mapper.BuilderContext(Settings.EMPTY, new ContentPath())));
    }

    @SneakyThrows
    public void testTypeParser_invalidDimension_thenThrows() {
        XContentBuilder mapping = XContentFactory.jsonBuilder()
            .startObject()
            .field(KNNConstants.TYPE, LateInteractionFieldMapper.CONTENT_TYPE)
            .field(KNNConstants.DIMENSION, 0)
            .endObject();

        expectThrows(
            IllegalArgumentException.class,
            () -> LateInteractionFieldMapper.PARSER.parse(FIELD_NAME, xContentBuilderToMap(mapping), buildParserContext())
        );
    }

    @SneakyThrows
    public void testParseCreateField_validMultiVector_thenEmitsLateInteractionField() {
        LateInteractionFieldMapper mapper = buildMapper(DIMENSION, SpaceType.INNER_PRODUCT);

        // two tokens of dimension 4
        float[][] expected = { { 0.1f, 0.2f, 0.3f, 0.4f }, { 0.5f, 0.6f, 0.7f, 0.8f } };
        XContentParser parser = multiVectorParser(expected);

        ParseContext.Document document = new ParseContext.Document();
        ParseContext parseContext = mockParseContext(parser, document);

        mapper.parseCreateField(parseContext);

        assertEquals(1, document.getFields().size());
        IndexableField field = document.getFields().get(0);
        assertTrue(field instanceof LateInteractionField);
        float[][] roundTrip = ((LateInteractionField) field).getValue();
        assertEquals(expected.length, roundTrip.length);
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(expected[i], roundTrip[i], 1e-6f);
        }
    }

    @SneakyThrows
    public void testParseCreateField_flatArray_thenThrows() {
        LateInteractionFieldMapper mapper = buildMapper(DIMENSION, SpaceType.INNER_PRODUCT);

        // a single flat vector [0.1,0.2,0.3,0.4] instead of a list-of-lists
        XContentBuilder b = XContentFactory.jsonBuilder().startObject();
        b.startArray(FIELD_NAME).value(0.1f).value(0.2f).value(0.3f).value(0.4f).endArray();
        b.endObject();
        XContentParser parser = createParserOnField(b);

        ParseContext parseContext = mockParseContext(parser, new ParseContext.Document());
        expectThrows(MapperParsingException.class, () -> mapper.parseCreateField(parseContext));
    }

    @SneakyThrows
    public void testParseCreateField_wrongInnerDimension_thenThrows() {
        LateInteractionFieldMapper mapper = buildMapper(DIMENSION, SpaceType.INNER_PRODUCT);

        // token of dimension 3 (expected 4)
        float[][] bad = { { 0.1f, 0.2f, 0.3f } };
        XContentParser parser = multiVectorParser(bad);

        ParseContext parseContext = mockParseContext(parser, new ParseContext.Document());
        expectThrows(MapperParsingException.class, () -> mapper.parseCreateField(parseContext));
    }

    // ---- helpers ----

    private LateInteractionFieldMapper buildMapper(int dimension, SpaceType spaceType) throws Exception {
        XContentBuilder mapping = XContentFactory.jsonBuilder()
            .startObject()
            .field(KNNConstants.TYPE, LateInteractionFieldMapper.CONTENT_TYPE)
            .field(KNNConstants.DIMENSION, dimension)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .endObject();
        LateInteractionFieldMapper.Builder builder = (LateInteractionFieldMapper.Builder) LateInteractionFieldMapper.PARSER.parse(
            FIELD_NAME,
            xContentBuilderToMap(mapping),
            buildParserContext()
        );
        return builder.build(new Mapper.BuilderContext(Settings.EMPTY, new ContentPath()));
    }

    private XContentParser multiVectorParser(float[][] vectors) throws Exception {
        XContentBuilder b = XContentFactory.jsonBuilder().startObject();
        b.startArray(FIELD_NAME);
        for (float[] v : vectors) {
            b.startArray();
            for (float f : v) {
                b.value(f);
            }
            b.endArray();
        }
        b.endArray();
        b.endObject();
        return createParserOnField(b);
    }

    /** Returns a parser positioned on the value token of FIELD_NAME, mimicking the mapper's entry state. */
    private XContentParser createParserOnField(XContentBuilder builder) throws Exception {
        XContentParser parser = createParser(builder);
        parser.nextToken(); // START_OBJECT
        parser.nextToken(); // FIELD_NAME
        parser.nextToken(); // value (START_ARRAY / VALUE_NULL)
        return parser;
    }

    private ParseContext mockParseContext(XContentParser parser, ParseContext.Document document) {
        IndexSettings indexSettingsMock = mock(IndexSettings.class);
        when(indexSettingsMock.getSettings()).thenReturn(Settings.EMPTY);
        ParseContext parseContext = mock(ParseContext.class);
        when(parseContext.doc()).thenReturn(document);
        when(parseContext.path()).thenReturn(new ContentPath());
        when(parseContext.parser()).thenReturn(parser);
        when(parseContext.indexSettings()).thenReturn(indexSettingsMock);
        return parseContext;
    }

    private Mapper.TypeParser.ParserContext buildParserContext() {
        IndexSettings indexSettings = new IndexSettings(
            IndexMetadata.builder("test")
                .settings(
                    Settings.builder()
                        .put(IndexMetadata.SETTING_VERSION_CREATED, CURRENT)
                        .put(IndexMetadata.SETTING_NUMBER_OF_SHARDS, 1)
                        .put(IndexMetadata.SETTING_NUMBER_OF_REPLICAS, 0)
                        .build()
                )
                .build(),
            Settings.EMPTY,
            new IndexScopedSettings(Settings.EMPTY, new HashSet<>(IndexScopedSettings.BUILT_IN_INDEX_SETTINGS))
        );
        MapperService mapperService = mock(MapperService.class);
        when(mapperService.getIndexSettings()).thenReturn(indexSettings);
        return new Mapper.TypeParser.ParserContext(
            null,
            mapperService,
            type -> LateInteractionFieldMapper.PARSER,
            Version.CURRENT,
            null,
            null,
            null
        );
    }
}

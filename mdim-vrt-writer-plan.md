# Plan: xarray → mdim VRT writer

## Goal

A pure-Python writer that takes an `xr.Dataset` plus an explicit source specification and emits a valid GDAL multidimensional VRT. The reader direction already exists via `gdx` (VRT → GDAL mdim → xarray); this is the missing half.

## Why pure Python

The writer demonstrates that VRT is a text spec, not a GDAL artefact. A writer that needs GDAL at runtime weakens that claim. `VirtualiZarr` is the precedent: it emits kerchunk / V3 manifest / Icechunk from a lazy xarray without calling the source-file libraries. A VRT writer is the same move, one level up.

Correctness is validated by round-trip against the reference reader (`osgeo.gdal` + `gdx`). If what we wrote opens and compares equal on `xr.testing.assert_identical`, modulo known-lossy bits, it's correct.

## Schema reference

Target XSD: GDAL `release/3.12` `gdalvrt.xsd` at <https://raw.githubusercontent.com/OSGeo/gdal/refs/heads/release/3.12/frmts/vrt/data/gdalvrt.xsd>.

Multidim content lives under a `<Group>` child of `<VRTDataset>`, not directly at the root:

```xml
<VRTDataset>
  <Group name="/">
    <Dimension .../>
    <Attribute .../>
    <Array ...>...</Array>
    <!-- nested <Group> is legal; we emit a single flat root group -->
  </Group>
</VRTDataset>
```

The writer pins to this XSD version. Any bump is an explicit plan revision.

## Scope

**In scope**

- Emit `<VRTDataset>` containing a single `<Group name="/">` with dimensions, indexing variables (coords), data variable arrays, source composition, attributes, CRS, and chunking hints
- Three composition modes: stack along a new dim, concat along an existing dim, spatial mosaic
- Inlined coord arrays (small dims like time, depth) and source-referenced coord arrays (large dims like lon/lat grids)
- Sources of any kind GDAL mdim can open: NetCDF, HDF5, Zarr, and nested mdim VRT

**Not in scope**

- Derived virtual arrays (GDAL expressions, pixel functions)
- Sub-file subsetting (see rules)
- Automatic introspection of dask graphs to infer sources — too fragile, require explicit sources
- Schema validation at write time — round-trip is the validation
- Anything the xarray data model doesn't represent natively — if it needs GDAL-specific semantics, it belongs in a hand-authored VRT

## Rules

### R1. Sources are referenced whole

Subsetting of a lazy source is permitted in principle, at the composition level. Sub-file subsetting is not.

The XSD provides four mechanisms that would let a `<Source>` present a transformed view of its underlying array. The writer emits **none** of them:

- `<SourceSlab offset count step>` — index-range subset of the source array. **Never emitted.**
- `<SourceView expr>` — expression-based view of the source. **Never emitted.**
- `<SourceTranspose>` — reorders source dims. **Never emitted** — the user must pre-align.
- `<DerivedArray>` with `<Step>` children (`View`, `Transpose`, `Resample`, `Grid`, `GetMask`, `GetUnscaled`) — computed arrays. **Never emitted.**

The writer does use `<DestSlab offset>`. This is *placement* of a whole source into the destination, not a subset: it says where in the destination this source goes. It is required for stack, concat, and mosaic (each source has to state its destination offset), and is not considered subsetting under this rule.

**Allowed**: dropping sources from a list, reordering sources, filtering participation by coord value, using a nested VRT as a source where the inner VRT describes a different composition of its own sources.

**Not allowed**: a spatial window, an index range, or any slice taken from within a single source along any dim. Each `<Source>` element references the source array in full.

If the user has `ds.isel(time=slice(5, 15))` where the slice crosses mid-file boundaries, the writer raises. Options for the user:

1. Re-chunk the input — materialize the slice into a new file or Zarr and write VRT against that
2. Keep the VRT at natural file boundaries and slice at read time
3. Build an outer VRT over pre-sliced intermediates

Rationale: keeping sources whole means the recipe is verifiable by reading sources directly, the XML stays compact, and composition lives at one clear conceptual level. Sub-file subsetting would turn each `<Source>` into a mini-slicing DSL — that's exactly what mdim VRT supports but not what we want the writer to model.

### R2. One composition mode per write call

Mixed composition (e.g. stack + mosaic in one `<Array>`) is expressible in VRT but not emitted by the writer directly. Compose it by writing nested VRTs, each with a single mode, and referencing them as sources of an outer VRT. Keeps each write call small, testable, and inspectable.

### R3. Source compatibility

All sources in a composition must agree on:

- dtype
- dims not participating in the composition (same names, same sizes)
- CRS (for mosaic; for stack/concat, CRS must be consistent if present at all)

The writer checks this up front. Mismatch raises with a diff.

### R4. CRS is explicit or detected, never silently inherited

If `crs=` is not passed and no CF or rio convention is detected on the Dataset, the writer raises. No silent "took the CRS from source 1" behaviour.

### R5. Monotonicity along concat dim

For concat composition, the coord values along the concat dim must be monotonic across sources (ascending or descending consistently). Otherwise the composition is ambiguous and the writer raises.

### R6. Sources are paths, not handles

The writer accepts filesystem paths, URLs, and GDAL-style VSI paths (`/vsicurl/...`, `/vsis3/...`). It does not accept opened file handles, open Datasets, or Zarr groups. The VRT references sources by identifier; nothing in-memory is part of the recipe.

### R7. Array names are globally unique

`<Array>` children of a VRT dataset share a namespace. Before writing, the writer checks for collisions between `data_vars` and `coords` and raises on collision. xarray doesn't require disjointness here but VRT does.

## Data model mapping

| xarray | mdim VRT |
|---|---|
| `Dataset` | `<VRTDataset><Group name="/">` — a single root group |
| `Dataset.dims` | `<Dimension name size indexingVariable>` in root Group; optional `type` and `direction` for CF axis roles when detected |
| `Dataset.coords[d]` where `d` is a dim name | `<Array name=d>` in root Group; values via `<RegularlySpacedValues>`, `<InlineValuesWithValueElement>`, or `<Source>` |
| `Dataset.data_vars[v]` | `<Array name=v>` in root Group with `<DimensionRef>` back-refs and one or more `<Source>` |
| `Dataset.attrs` | `<Attribute>` children of root `<Group>` |
| `DataArray.attrs` | `<Attribute>` children of the corresponding `<Array>` |
| Detected or passed CRS | `<SRS dataAxisToSRSAxisMapping>` inside `<Array>`, WKT in content |
| `.dtype` | `<DataType>` child element of `<Array>` — element, not attribute (mdim differs from classic raster VRT here) |
| `.chunks` | `<BlockSize>` space-separated integer list, one per dim in array-dim order |
| source composition | one `<Source>` per input, each with `<SourceFilename>`, `<SourceArray>`, and `<DestSlab offset>` (always emitted, zeros included, for consistency) |

### Coord value emission

Three carriers, dispatched by coord shape:

- **`<RegularlySpacedValues start increment>`** when uniform spacing is detected within tolerance. Covers daily / monthly time axes, regular depth levels, regular lat/lon.
- **`<InlineValuesWithValueElement>`** with one `<Value>` per element, when the coord is below the `coord_mode="auto"` size threshold (default ~10_000) and not regularly spaced.
- **`<Source>`** pointing at a coord variable in one of the source files, when the coord is above threshold (typical case: 2D lon/lat grids from curvilinear datasets).

The XSD also allows `<ConstantValue>` and `<InlineValues>` (space-separated string form). The writer does not emit either — `ConstantValue` is rarely what you want for a coord, and `InlineValuesWithValueElement` is more explicit than the string form at comparable verbosity.

## Schema conformance

The writer targets a deliberate subset of the XSD. Emitted elements:

- `<VRTDataset>` with no `subClass` attribute — the mdim case is structurally distinct from Warped / Pansharpened / Processed subclasses
- `<Group name="/">` — a single root group, flat
- `<Dimension name size indexingVariable type? direction?>`
- `<Array name>` containing `<DataType>`, `<DimensionRef>` (or `<Dimension>` for coord arrays), `<BlockSize>?`, `<SRS>?`, `<Unit>?`, `<NoDataValue>?`, `<Offset>?`, `<Scale>?`, a value carrier, `<Attribute>*`
- `<SRS dataAxisToSRSAxisMapping?>` with WKT content, inside `<Array>`
- `<Attribute name>` with `<DataType>` and `<Value>*` at Group or Array level
- `<Source>` with `<SourceFilename relativeToVRT?>`, `<SourceArray>`, `<DestSlab offset>`
- Coord-value carriers: `<RegularlySpacedValues>`, `<InlineValuesWithValueElement>`, `<Source>`

Elements the writer does **not** emit (with reasons):

- Classic raster VRT elements (`<VRTRasterBand>`, `<SimpleSource>`, `<ComplexSource>`, `<KernelFilteredSource>`, etc.) — out of scope
- `<DerivedArray>` and its `<Step>` children (`<View>`, `<Transpose>`, `<Resample>`, `<Grid>`, `<GetMask>`, `<GetUnscaled>`) — R1
- `<SingleSourceArray>` — the short form for trivial cases; we always emit the full `<Array>` for uniformity
- `<SourceSlab>`, `<SourceView>`, `<SourceTranspose>` — R1
- `<ConstantValue>`, `<InlineValues>` (string form) — we prefer `<InlineValuesWithValueElement>`
- `<GDALWarpOptions>`, `<PansharpeningOptions>`, `<ProcessingSteps>`, `<Input>`, `<OutputBands>` — subClass features of classic VRT, not used in mdim

The test suite XSD-validates every emitted file against the pinned schema (via `lxml` as a test-only dep — see Dependencies). Any addition to the emitted element set is a plan revision, not an implementation detail.

## API

### v0 — minimal viable writer

```python
write_mdim_vrt(
    ds,                   # xr.Dataset
    sources,              # list[str | Path] or list[dict] with per-source metadata
    path,                 # output .vrt path
    *,
    composition,          # "stack" | "concat" | "mosaic"
    concat_dim=None,      # required for "stack" and "concat"
    crs=None,             # explicit CRS; else detect via cf/rio; else raise
    block_size=None,      # override; else from ds.chunks if present; else omit
    coord_mode="auto",    # "inline" | "source" | "auto" (size threshold) | dict per coord
)
```

Explicit sources, single composition mode, no derived expressions. Covers the BRAN2023 demo completely.

### v1 — coord source resolution

Default `coord_mode="auto"`: inline coords for dim sizes below a threshold (e.g. 10_000), source-reference otherwise. Per-coord override via dict. No code surface beyond `_emit_coord_array()` dispatching on mode.

### v2 — Zarr as source

A Zarr store is a source like any other once GDAL mdim opens it. The demo is: open an existing Zarr with xarray, call `write_mdim_vrt` with the Zarr path as the source, round-trip back. Demonstrates "recipe recoverable from rendering". Mostly test surface, no new code beyond source-kind dispatch.

### v3 — nested VRT sources

A source whose path ends `.vrt` is a first-class source kind. Falls out for free from source-kind dispatch — the writer doesn't care what the source is, only that mdim can open it. Demo: stack a collection of VRTs under an outer VRT.

### Optional accessor layer

```python
@xr.register_dataset_accessor("vrt")
class VrtAccessor:
    def from_sources(self, sources, composition, concat_dim=None): ...
    def write(self, path, **opts): ...
```

`ds.vrt.from_sources(...)` attaches composition metadata to the Dataset; `ds.vrt.write(path)` emits. Dataset stays a plain `xr.Dataset` everywhere else. Mirrors the `rioxarray` pattern.

## Dependencies

- `xarray` (required)
- `numpy` (required)
- `xml.etree.ElementTree` from stdlib for emission, with a small `_indent()` helper
- `cf_xarray` *optional*, only for CRS auto-detection; explicit `crs=` works without it

No `lxml`, no `gdal`, no `rasterio` in the writer itself. All three may appear in the test suite.

## Validation strategy

Round-trip is the correctness criterion.

1. Construct a known `xr.Dataset` from N explicit source files
2. Write VRT with the writer
3. Open the VRT with `osgeo.gdal` mdim API (structural check) and with `gdx` (xarray-level check)
4. Compare to the original with `xr.testing.assert_identical`, modulo known-lossy bits (dask graph identity, non-deterministic attr ordering)

Cases:

- BRAN2023 stack, 96 NetCDF, time-stacked — the blog-post demo, end-to-end
- Two synthetic files concatenated along an existing dim
- Three synthetic files mosaicked spatially
- A Zarr re-described as VRT — the ESA demo
- A VRT-of-VRTs — the nested case
- Negative tests for each rule: sub-file-subset request, mixed composition, dtype mismatch, CRS missing, concat-dim non-monotonic, array-name collision, opened-handle source. Each raises with a clear message.

## R parallel

`vrtstack` gains `write_mdim_vrt()` alongside the existing `gdal_mdim_stack()`. `gdal_mdim_stack()` is the GDAL-based path (shells to the CLI or writes via the GDAL mdim API via `gdalraster`); `write_mdim_vrt()` is the without-GDAL path, pure R emitting the same XML. Both paths are tested against `gdalraster` multidim reads. Convergence criterion: the two writers produce output that opens to structurally identical datasets.

Same rules apply. Same data-model mapping, modulo R-side coord conventions (`stars` proxies, `dimensions` attributes). The R and Python writers are independent implementations of the same spec, which is the whole point.

## Open questions

- **Chunking default when `.chunks` is absent.** Infer from source files (requires opening one) or omit? Proposed: omit. User passes `block_size=` if they want it in the VRT.
- **Attribute fidelity.** CF attrs round-trip cleanly; arbitrary user attrs may not fit VRT `<Attribute>` types. Filter silently, warn, or raise? Proposed: warn, with filtered keys listed.
- **Endianness and fill values.** Carried by source file conventions and also expressible at the VRT level. Proposed: trust source-file provenance, only emit explicit values when the user overrides.
- **Empty coord arrays.** A dim with no coord variable — legal in xarray, less clear in VRT. Proposed: emit `<Dimension>` without `indexingVariable`, don't emit an `<Array>`.
- **Mosaic with overlap.** Ambiguous pixel values in overlap regions. Proposed: v0 raises on detected overlap; v1 adds an explicit overlap-resolution policy arg.

## Non-answers (for later)

- **Pure reader** (VRT → xarray without GDAL). Worthwhile eventually; not needed for the thesis. The writer alone proves the spec is portable.
- **Direct VRT → kerchunk / Icechunk** path without going through GDAL. Same category — valuable, not required now.

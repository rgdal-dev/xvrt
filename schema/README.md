# Pinned schema

The writer targets `gdalvrt.xsd` from the GDAL `release/3.12` branch:

    https://raw.githubusercontent.com/OSGeo/gdal/refs/heads/release/3.12/frmts/vrt/data/gdalvrt.xsd

Drop the file here as `gdalvrt.xsd`. The test suite validates every emitted
file against it; the writer itself doesn't read the schema.

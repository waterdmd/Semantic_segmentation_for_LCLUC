import os
import glob
from osgeo import gdal

def merge_rasters(input_directory, output_path):
    # Get a list of all the raster files in the directory
    raster_files = glob.glob(os.path.join(input_directory, "*.tif"))

    if not raster_files:
        raise ValueError(f"No raster files found in directory {input_directory}")

    # Options for building VRT
    vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest', addAlpha=True)
    vrt_path = os.path.join(input_directory, "temp.vrt")

    # Create a VRT (Virtual Dataset) from the input rasters
    vrt = gdal.BuildVRT(vrt_path, raster_files, options=vrt_options)
    if vrt is None:
        raise RuntimeError("Failed to build VRT")

    # Translate the VRT to a single merged GeoTIFF
    gdal.Translate(output_path, vrt, format="GTiff")
    vrt = None  # Close the VRT to flush to disk

    # Clean up temporary VRT file
    if os.path.exists(vrt_path):
        os.remove(vrt_path)

    print(f"Successfully merged rasters into {output_path}")

# Example usage
input_directory = "/data/Predictions_MANET"
output_path = "/data/composite_prediction_2023_MANET_T.tif"
merge_rasters(input_directory, output_path)

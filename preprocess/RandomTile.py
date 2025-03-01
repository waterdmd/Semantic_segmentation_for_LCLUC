import rasterio
import numpy as np
import os
import glob as glob
import random
import sys

def extract_and_save_window(image_path, row, col, class_value, size=128, fill_value=0.0001, nan_fill_value=0.0001, class_fill_value=0, out_of_bounds_class=0):
# def extract_and_save_window(image_path, row, col, class_value, size=64, fill_value=0.0001, nan_fill_value=0.0001, class_fill_value=8, out_of_bounds_class=8):
    window_size = size // 2
    
    output_dir = os.path.dirname(image_path)+'/val/'+os.path.splitext(os.path.basename(image_path))[0]+'/'
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(image_path) as src:
        # Initialize arrays to hold the window data with out-of-bound values
        window_data = np.full((src.count, size, size), fill_value, dtype=src.dtypes[0])
        window_data[-1, :, :] = out_of_bounds_class  # Set the class band to the out of bounds class value

        # Calculate bounds of the window within the image boundaries
        row_start = max(row - window_size, 0)
        row_end = min(row + window_size, src.height)
        col_start = max(col - window_size, 0)
        col_end = min(col + window_size, src.width)

        # Define the window to read the actual image data
        read_window = rasterio.windows.Window.from_slices((row_start, row_end), (col_start, col_end))
        read_data = src.read(window=read_window)

        # Place the read data into the window array
        dest_row_start = row_start - (row - window_size)
        dest_row_end = dest_row_start + (row_end - row_start)
        dest_col_start = col_start - (col - window_size)
        dest_col_end = dest_col_start + (col_end - col_start)
        window_data[:, dest_row_start:dest_row_end, dest_col_start:dest_col_end] = read_data

        # Handle NaN values in in-boundary data: set them to nan_fill_value and class to class_fill_value
        for b in range(src.count - 1):  # Exclude the last class band
            nan_mask = np.isnan(window_data[b, dest_row_start:dest_row_end, dest_col_start:dest_col_end])
            window_data[b, dest_row_start:dest_row_end, dest_col_start:dest_col_end][nan_mask] = nan_fill_value
        window_data[-1, dest_row_start:dest_row_end, dest_col_start:dest_col_end][nan_mask] = class_fill_value

        # Update metadata for output file
        out_meta = src.meta.copy()
        out_meta.update({
            "height": size,
            "width": size,
            "transform": rasterio.windows.transform(read_window, src.transform)
        })

        # Save the output file
        output_file = os.path.join(output_dir, f'class_{class_value}_{row}_{col}.tif')
        with rasterio.open(output_file, 'w', **out_meta) as out_raster:
            out_raster.write(window_data)
        print(f"Saved: {output_file}")
        return True
    
    
    
    
    
    


def process_image(image_path, target_tiles=10, ignorelist = None):
    
    for images in glob.glob(image_path + '*.tif'):
        print(f"Processing image: {images}")
        with rasterio.open(images) as src:
            image = src.read()
            class_band = image[-1]  # Last band contains class information
            classes = np.unique(class_band)
            # classes  = [0,1,2,4,5,6,7,8]

            for class_value in classes:
                if class_value not in ignorelist:
                    print(f"Processing class {class_value}")
                    indices = np.column_stack(np.where(class_band == class_value))
                    
                    tile_count = 0
                    consecutive_failures = 0
                    used_points = set()

                    while tile_count < target_tiles and len(indices) > 0:
                        # Select a random index
                        rand_index = random.randint(0, len(indices) - 1)
                        row, col = indices[rand_index]
                        
                        # Remove the selected index from the list
                        indices = np.delete(indices, rand_index, axis=0)

                        if (row, col) in used_points:
                            continue

                        if extract_and_save_window(image_path=images, row=row, col=col, class_value=class_value):
                            tile_count += 1
                            used_points.add((row, col))
                        else:
                            consecutive_failures += 1
                            if consecutive_failures >= 100:
                                print(f"Stopped processing class {class_value} after 100 consecutive failures.")
                                break

                    if tile_count < target_tiles:
                        print(f"Warning: Only {tile_count} valid tiles found for class {class_value}.")

    
    
# def process_image(image_path, target_tiles=10):
    
#     for images in glob.glob(image_path + '*.tif'):
#         print(f"Processing image: {images}")
#         with rasterio.open(images) as src:
#             image = src.read()
#             class_band = image[-1]  # Last band contains class information
#             classes = np.unique(class_band)

#             for class_value in classes:
#                 if class_value != 8:
#                     print(f"Processing class {class_value}")
#                     indices = np.column_stack(np.where(class_band == class_value))
                    
#                     # Divide indices into smaller subsets
#                     num_subsets = min(len(indices), target_tiles)  # Ensure at least one point per subset
#                     subset_size = len(indices) // num_subsets
#                     subset_indices = [indices[i:i+subset_size] for i in range(0, len(indices), subset_size)]
                    
#                     # Shuffle each subset independently
#                     for subset in subset_indices:
#                         random.shuffle(subset)
                    
#                     # Merge shuffled subsets
#                     indices = np.vstack(subset_indices)
#                     random.shuffle(indices)  # Shuffle the merged indices
                                        
#                     tile_count = 0
#                     consecutive_failures = 0
#                     used_points = set()

#                     for row, col in indices:
#                         if (row, col) in used_points:
#                             continue

#                         if extract_and_save_window(image_path=images, row=row, col=col, class_value=class_value):
#                             tile_count += 1
#                             used_points.add((row, col))
#                             if tile_count == target_tiles:
#                                 break
#                         else:
#                             consecutive_failures += 1
#                             if consecutive_failures >= 100:
#                                 print(f"Stopped processing class {class_value} after 100 consecutive failures.")
#                                 break

#                     if tile_count < target_tiles:
#                         print(f"Warning: Only {tile_count} valid tiles found for class {class_value}.")
    
    
    
    
# def process_image(image_path, target_tiles=10):
    
#     for images in glob.glob(image_path + '*.tif'):
#         print(f"Processing image: {images}")
#         with rasterio.open(images) as src:
#             image = src.read()
#             class_band = image[-1]  # Last band contains class information
#             classes = np.unique(class_band)

#             for class_value in classes:
#                 if class_value != 8:
#                     print(f"Processing class {class_value}")
#                     indices = np.column_stack(np.where(class_band == class_value))
#                     random.shuffle(indices)
#                     random.shuffle(indices)
#                     tile_count = 0
#                     consecutive_failures = 0
#                     used_points = set()

#                     for row, col in indices:
#                         if (row, col) in used_points:
#                             continue

#                         if extract_and_save_window(image_path=images, row=row, col=col, class_value=class_value):
#                             tile_count += 1
#                             used_points.add((row, col))
#                             if tile_count == target_tiles:
#                                 break
#                         else:
#                             consecutive_failures += 1
#                             if consecutive_failures >= 100:
#                                 print(f"Stopped processing class {class_value} after 100 consecutive failures.")
#                                 break

#                     if tile_count < target_tiles:
#                         print(f"Warning: Only {tile_count} valid tiles found for class {class_value}.")
                        
                    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tiling_script.py /path/to/your/images/")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    # Ensure the path ends with a slash
    if not image_dir.endswith('/'):
        image_dir += '/'
    
    process_image(image_dir, target_tiles=500, ignorelist=[])               

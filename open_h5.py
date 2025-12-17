import h5py
import rasterio
from rasterio.transform import from_origin

h5_file = "GW1AM2_20250300_01M_EQMA_L3SGSMCHF3300300.h5"
with h5py.File(h5_file, "r") as f:
    # Duyệt tất cả các group và dataset
    def print_h5(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"[DATASET] {name}  shape={obj.shape}  dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"[GROUP]   {name}")
    
    f.visititems(print_h5)

# with h5py.File(h5_file, "r") as f:
#     sm = f["Geophysical_Data/sm_surface"][:]              # soil moisture surface
#     soil_temp = f["Geophysical_Data/soil_temp_layer1"][:] # soil temp layer 1
#     lat = f["cell_lat"][:]
#     lon = f["cell_lon"][:]

# # --- tạo transform ---
# res_lat = abs(lat[1,0] - lat[0,0])
# res_lon = abs(lon[0,1] - lon[0,0])
# min_lon, max_lat = lon.min(), lat.max()
# transform = from_origin(min_lon, max_lat, res_lon, res_lat)

# # --- lưu soil moisture ---
# out_tif_sm = "soil_moisture_surface.tif"
# with rasterio.open(
#     out_tif_sm, "w",
#     driver="GTiff",
#     height=sm.shape[0],
#     width=sm.shape[1],
#     count=1,
#     dtype=sm.dtype,
#     crs="EPSG:4326",
#     transform=transform
# ) as dst:
#     dst.write(sm, 1)

# # --- lưu soil temp ---
# out_tif_temp = "soil_temp_layer1.tif"
# with rasterio.open(
#     out_tif_temp, "w",
#     driver="GTiff",
#     height=soil_temp.shape[0],
#     width=soil_temp.shape[1],
#     count=1,
#     dtype=soil_temp.dtype,
#     crs="EPSG:4326",
#     transform=transform
# ) as dst:
#     dst.write(soil_temp, 1)

# print("Xuất thành công:", out_tif_sm, out_tif_temp)

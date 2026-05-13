import ee
import torch
torch.set_num_threads(1)
from tqdm import tqdm
import argparse

ee.Authenticate()
ee.Initialize(project='project5324-448512')

torch.set_num_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gedi",
    action="store_true",
    help="Export GEDI (LiDAR) tiles in addition to the other exports"
)
args = parser.parse_args()

gedi_activated = args.gedi

# ----------------------------------------------------
# Get all current tasks
# ----------------------------------------------------
tasks = ee.data.listOperations()

my_tasks = [
    t for t in tasks
    if "img_tile_" in t.get("metadata", {}).get("description", "")
]

# ----------------------------------------------------
# Amazon Basin (simplified)
# ----------------------------------------------------
amazon = ee.Geometry.Polygon([
    [[-60.427, -2.798],
     [-60.427, -2.372],
     [-59.764, -2.372],
     [-59.764, -2.798],
     [-60.427, -2.798]]
])

# ----------------------------------------------------
# Time Period
# ----------------------------------------------------
startDate = ee.Date.fromYMD(2020, 1, 1)
endDate = startDate.advance(1, "year")

# ----------------------------------------------------
# Sentinel-1 Processing (VV, VH)
# ----------------------------------------------------
s1 = (
    ee.ImageCollection("COPERNICUS/S1_GRD")
    .filterBounds(amazon)
    .filterDate("2021-01-01", "2022-01-01")
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    .filter(ee.Filter.eq("instrumentMode", "IW"))
    .select(["VV", "VH"])
    .median()
)

# ----------------------------------------------------
# Cloud Masking Function for Sentinel-2
# ----------------------------------------------------
def maskS2clouds(image):
    qa = image.select("QA60")

    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11

    mask = (
        qa.bitwiseAnd(cloudBitMask).eq(0)
        .And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    ) # Mask imagery with the QA60 quality-assurance band from S2 /!\ QA60 does not mean remove >= 60% cloudy

    return image.updateMask(mask).divide(10000)

# ----------------------------------------------------
# Harmonized Sentinel-2 Collection with Cloud Mask
# ----------------------------------------------------
datasets2 = (
    ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    .filterBounds(amazon)
    .filterDate("2021-01-01", "2022-01-01")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)) # <= 20% cloudy/cirrus imagery only
    .map(maskS2clouds)
    .select(["B5", "B6", "B8", "B8A"]) # Are those all relevant? Are there superfluous bands, or missing bands?
    .median()
)

datasets2 = datasets2.clip(amazon)

# ----------------------------------------------------
# Stack All Bands
# ----------------------------------------------------

# First step of label construction for Copernicus-FM

stackimage = (
    s1.toFloat()
    .addBands(datasets2.toFloat())
    .clip(amazon)
)

def compute_tile_label_from_stack(stackimage, tile_geom, forest_frac_thresh=0.00):
    """
    Returns 0/1 label for a tile to test for forest presence:
      NDVI > 0.6, NDRE > 0.3, SAR_ratio > 0.25
    computed pixel-wise then averaged over the tile.
    """
    VV = stackimage.select("VV")
    VH = stackimage.select("VH")
    B5 = stackimage.select("B5")
    B6 = stackimage.select("B6")
    B8 = stackimage.select("B8")
    B8A = stackimage.select("B8A")

    eps = 1e-6

    ndvi = B8.subtract(B5).divide(B8.add(B5).add(eps))
    ndre = B8A.subtract(B6).divide(B8A.add(B6).add(eps))
    sar_ratio = VH.divide(VV.add(eps))

    forest_mask = ndvi.gt(0.6).And(ndre.gt(0.3)).And(sar_ratio.gt(0.25))

    frac = forest_mask.rename("forest").reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=tile_geom,
        scale=10,
        maxPixels=1e8,
        bestEffort=True
    ).get("forest")

    return ee.Number(frac).gte(forest_frac_thresh)

# ----------------------------------------------------
# ESA CCI Above Ground Biomass
# ----------------------------------------------------

# "GT" biomass density data in the selected tiles

agb = ee.ImageCollection("projects/sat-io/open-datasets/ESA/ESA_CCI_AGB")

agb_2021 = (
    agb.filterDate("2021-01-01", "2022-01-01")
    .first()
    .select(["AGB"])
    .clip(amazon)
)

# ----------------------------------------------------
# GEDI L4A AGB Data
# ----------------------------------------------------

# /!\ Copernicus-FM currently does not support LiDAR data (23/04/2026);
# /!\ This issue could be solved by using a LiDAR data embedder, then fusing the Copernicus-FM feature with the LiDAR feature;
# Therefore, by default, GEDI data is not downloaded, seeing as it is currently useless.
# Activate the --gedi flag to download and stack GEDI data to the model's input tensor.

if gedi_activated:
    targetScale = 100 # GEDI is pretty coarse

    gedi = (
        ee.ImageCollection("LARSE/GEDI/GEDI04_A_002_MONTHLY")
        .filterBounds(amazon)
    )

    def qualityMask(image):
        return (
            image.updateMask(image.select("l4_quality_flag").eq(1))
            .updateMask(image.select("degrade_flag").eq(0))
        )

    def errorMask(image):
        relative_se = image.select("agbd_se").divide(image.select("agbd"))
        return image.updateMask(relative_se.lte(0.5))

    gediProcessed = gedi.map(qualityMask).map(errorMask)

    gediMosaic = (
        gediProcessed
        .mosaic()
        .reproject(crs="EPSG:4326", scale=targetScale)
        .reduceResolution(
            reducer=ee.Reducer.median(),
            maxPixels=1024
        )
    )

    gediMosaic = gediMosaic.select(["agbd", "delta_time"])

    # ----------------------------------------------------
    # Convert GEDI delta_time to Actual Timestamp
    # ----------------------------------------------------
    deltaTimeImg = gediMosaic.select("delta_time")

    meanDelta = deltaTimeImg.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=amazon,
        scale=25,
        maxPixels=1e9
    ).get("delta_time")

    millisSinceEpoch = ee.Number(meanDelta).divide(1000)
    gediEpochMillis = ee.Date("2018-01-01T00:00:00Z").millis()
    actualDateMillis = gediEpochMillis.add(millisSinceEpoch)
    actualDate = ee.Date(actualDateMillis)

    print("Actual Date:", actualDate.format("YYYY-MM-dd HH:mm:ss").getInfo())

    gediMosaic = gediMosaic.set({
        "system_time_start": actualDate.millis(),
        "timestamp_utc": actualDate.format("YYYY-MM-dd HH:mm:ss")
    })

    # ----------------------------------------------------
    # Final Masked GEDI AGB
    # ----------------------------------------------------
    gediMosaic = (
        gediMosaic.select("agbd")
        .updateMask(gediMosaic.select("agbd").lte(500))
        .clip(amazon)
    )

    stats = gediMosaic.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=amazon,
        scale=25,
        maxPixels=1e13,
        bestEffort=True
    )

    print("GEDI AGBD min & max (masked):", stats.getInfo())

# ----------------------------------------------------
# Create Grid Tiles
# ----------------------------------------------------
projection = ee.Projection("EPSG:4326").atScale(2640)
grid = amazon.coveringGrid(projection)
print("Grid size:", grid.size().getInfo()) # At this scale, 504 patches of size [1, 265, 265] (CHW)
tiles = grid.toList(2000)

# ----------------------------------------------------
# Export Random Tiles (Equivalent Loop)
# ----------------------------------------------------
labels = []

for i in tqdm(range(0, 503)): # EPSG:4326, 2640 at scale 10m makes 504 tiles
    tile = ee.Feature(tiles.get(i)).geometry()

    # label (0/1) computed server-side, fetched as Python int
    label_i = compute_tile_label_from_stack(stackimage, tile).getInfo()
    labels.append(int(label_i))

    stack_patch = stackimage.clip(tile)
    if gedi_activated:
        agb_patch = gediMosaic.clip(tile)
    esa_agb_patch = agb_2021.clip(tile)

    ee.batch.Export.image.toAsset(
        image=stack_patch,
        description="img_tile_" + str(i),
        assetId="projects/project5324-448512/assets/FM_Patches/04022026v2/img_tile_" + str(i),
        region=tile.bounds(),
        scale=10,
        maxPixels=1e8
    ).start()

    if gedi_activated:
        ee.batch.Export.image.toAsset(
            image=agb_patch,
            description="gedi_tile_" + str(i),
            assetId="projects/project5324-448512/assets/FM_Patches/04022026v2/gedi_tile_" + str(i),
            region=tile.bounds(),
            scale=10,
            maxPixels=1e8
        ).start()

    ee.batch.Export.image.toAsset(
        image=esa_agb_patch,
        description="esa_tile_" + str(i),
        assetId="projects/project5324-448512/assets/FM_Patches/04022026v2/esa_tile_" + str(i),
        region=tile.bounds(),
        scale=10,
        maxPixels=1e8
    ).start()

tile_labels = torch.tensor(labels, dtype=torch.long)
torch.save(tile_labels, "tile_labels.pt") # Required for actual tile downloading later on
print("Saved tile_labels.pt with shape:", tile_labels.shape)
print("Class counts:", torch.bincount(tile_labels))

# Helper functions for the computation of tiles

def get_tile_dict(index):
    image_asset_id = f'projects/project5324-448512/assets/FM_Patches/04022026v2/img_tile_{index}'
    mask_asset_id = f'projects/project5324-448512/assets/FM_Patches/04022026v2/esa_tile_{index}'

    # GEE image and mask loading
    image = ee.Image(image_asset_id)
    mask = ee.Image(mask_asset_id)

    # Metadata (centroid lat, lon, timestamp)
    geom = image.geometry().centroid()
    latlon = geom.coordinates().getInfo()

    # Continuous data contains no information about time

    """timestamp = mask.get('timestamp_utc').getInfo()
    start = datetime(1970, 1, 1)
    date_obj = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")"""

    # Get scale from the first band projection
    scale = image.projection().nominalScale().getInfo()

    # Get region (geometry)
    region = image.geometry()
    bounds = region.bounds()
    region = ee.Geometry.Polygon(bounds.getInfo()['coordinates'][0])

    # Return input tensor for CopernicusFM
    return {
        'image': image,
        'mask': mask,
        'lat': latlon[1],
        'lon': latlon[0],
        'time': None, #(date_obj - start).days,
        'scale': scale,
        'region': region
    }

# There are 504 patches in the currently downloaded data
# Dataset curation

# Convert input to tensors, for Copernicus-FM

def convert_to_tensor_dict(tile_dict,  mean_img=None, std_img=None, mean_msk=None, std_msk=None):
    region = tile_dict['region']
    scale = tile_dict['scale']

    # Sample the mask band values in the region
    mask_sample = tile_dict['image'].mask().sample(region=region, scale=scale, numPixels=1)
    mask_values = mask_sample.getInfo()

    # Convert image (6 bands) and mask to np arrays
    image_arr = geemap.ee_to_numpy(tile_dict['image'], region=region, scale=scale)
    mask_arr = geemap.ee_to_numpy(tile_dict['mask'], region=region, scale=scale)


    # Clean up the data
    image_arr = np.where(np.isneginf(image_arr), 0, image_arr)
    image_arr = np.nan_to_num(image_arr, nan=0.0, posinf=0.0, neginf=0.0)

    mask_arr = np.where(np.isneginf(mask_arr), 0, mask_arr)
    mask_arr = np.nan_to_num(mask_arr, nan=0.0, posinf=0.0, neginf=0.0)

    img_processed = np.moveaxis(image_arr, -1, 0)
    mask_processed = mask_arr.squeeze() # [H, W]
    mask_processed = np.newaxis(np.newaxis(mask_processed)) # [1, H, W]

    image_tensor = torch.tensor(img_processed, dtype=torch.float32)

    # per-channel z-normalization over H,W
    mean = image_tensor.mean(dim=(1, 2), keepdim=True)
    std  = image_tensor.std(dim=(1, 2), keepdim=True)
    image_tensor = (image_tensor - mean) / (std + 1e-6)
    
    mask_tensor = torch.tensor(mask_processed).float()

    lat = torch.tensor(tile_dict['lat']).float()
    lon = torch.tensor(tile_dict['lon']).float()

    return {
        'image': image_tensor,
        'mask': mask_tensor,
        'lat': lat,
        'lon': lon,
        'time': torch.tensor([0]), # No time delta anywhere
        'scale': scale
    }

# Applies the "non-forest" threshold to every downloaded tiles
print("Computing stratification labels (forest vs non-forest)...")

def precompute_stratification_labels(tile_indices, out_path="tile_labels.pt"):
    labels = torch.zeros(len(tile_indices), dtype=torch.long)

    for i, tile_id in enumerate(tqdm(tile_indices, desc="Computing strat labels")):
        tile_dict = get_tile_dict(tile_id)
        tensor_dict = convert_to_tensor_dict(tile_dict)

        # IMPORTANT: compute NDVI on RAW image BEFORE normalization
        img_raw = torch.tensor(
            np.moveaxis(
                geemap.ee_to_numpy(
                    tile_dict["image"],
                    region=tile_dict["region"],
                    scale=tile_dict["scale"]
                ),
                -1, 0
            ),
            dtype=torch.float32
        )

        labels[i] = tile_indices[i]

    torch.save(labels, out_path)
    print(f"Saved stratification labels → {out_path}")
    return tile_dict, tensor_dict

# ----------------------------------------------------
# Get all current tasks (again)
# ----------------------------------------------------
tasks = ee.data.listOperations()

my_tasks = [
    t for t in tasks
    if "img_tile_" in t.get("metadata", {}).get("description", "")
]
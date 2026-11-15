// Sanjay: export one Sentinel-1 VV+VH tile over Silchar (Assam floods, June 2022).
// Paste into https://code.earthengine.google.com (free for non-commercial use; register once) -> Run
// -> Tasks tab -> Run both exports -> files land in your Google Drive.
// Output matches Sen1Floods11's format (GEE S1_GRD, dB, EPSG:4326, 10 m, ~512 x 512 px), so the trained
// models see the same preprocessing they were trained on.

var centre = ee.Geometry.Point([92.80, 24.82]);   // Silchar, Cachar district. Move if the tile looks dry.
var aoi = centre.buffer(2560).bounds();           // ~5 km square

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(aoi)
  .filterDate('2022-06-17', '2022-07-05')         // flood peak window; widen if nothing is listed
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .sort('system:time_start');

print('Acquisitions (copy the one you export into the upload form):',
  s1.aggregate_array('system:time_start').map(function (t) { return ee.Date(t).format(); }));

var img = ee.Image(s1.first()).select(['VV', 'VH']);   // or s1.toList(10).get(1) for the 2nd date
Map.centerObject(aoi, 13);
Map.addLayer(img.select('VV'), {min: -25, max: 0}, 'VV (dark = water)');

Export.image.toDrive({image: img.toFloat().clip(aoi), description: 'assam_silchar_2022_s1',
  region: aoi, scale: 10, crs: 'EPSG:4326', fileFormat: 'GeoTIFF'});

// Permanent water (JRC, water >= 50% of the time 1984-2021): Sanjay excludes it from "flood".
var perm = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').unmask(0).gte(50).toByte();
Export.image.toDrive({image: perm.clip(aoi), description: 'assam_silchar_2022_perm',
  region: aoi, scale: 10, crs: 'EPSG:4326', fileFormat: 'GeoTIFF'});

# Judge samples: real Sentinel-1 images to upload into Sanjay

Upload one `.tif` with "+ Add a Sentinel-1 scene". Optionally add its matching `_permanent-water.tif` so permanent rivers are not counted as flood. Each file is a 2-band GeoTIFF (VV, VH, decibels, WGS84), about 5 km square, under 2 MB. Source: Sen1Floods11 v1.1 (Bonafilia et al., CVPRW 2020), CC BY 4.0.

| File | What it is | Seen in training? | What Sanjay does (onboard model, checked 16 Nov) |
|---|---|---|---|
| 1_unseen-Assam-2016_A | Assam, 12 Aug 2016, centre 26.745 N, 93.758 E | No (held-out India test set) | Alert. About 6.4 km² flooded with the permanent-water file, 11.5 km² without it (the river then counts as flood). Hand labels: 47% water. |
| 2_unseen-Assam-2016_B | Assam, 12 Aug 2016, centre 26.469 N, 93.298 E | No (held-out India test set) | No alert. Hand labels: 0% water. |
| 3_Spain-2019-calibration-image | Spain, 17 Sep 2019, centre 38.152 N, 0.713 W | Not trained on; used only to calibrate the 90% interval and tune the alert rule | Alert. About 6.7 km² flooded with the permanent-water file. Hand labels: 37% water. |

Notes for whoever runs the demo:
- The app shows no accuracy check for uploads, because it does not receive hand labels. For reference, the onboard model's overlap with the expert labels (IoU) is 0.76 on file 1 and 0.72 on file 3.
- File names do not reveal the expected result, so a judge can pick blind.
- Uploading a normal photo, JPG or optical image should be rejected with a clear message. That is correct behaviour.

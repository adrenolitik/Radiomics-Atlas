# Radiomics Atlas
This dataset provides radiomics features extracted from a total 53 organs and anatomical structures (including parenchymal organs, vessels, muscles, and bones) depicted in abdominal and pelvic CT scans of 531 individuals. Radiomics data have been extracted from contrast-enhanced and non-contrast-enhanced images. It is publicly available and is intended for radiomics research applications.


This dataset can be used under the License CC-BY-NC-ND.

If you use this dataset, please cite:
Kapetanou, E., Malamas, S., Leventis, D.,  Karantanas, A.H., Klontzas, M.E., et al. Developing a Radiomics Atlas Dataset of normal Abdominal and Pelvic computed Tomography (RADAPT). J Digit Imaging. Inform. med. 37, 1273–1281 (2024). https://doi.org/10.1007/s10278-024-01028-7 

## Pancreas and adrenal glands: public CT segmentation datasets

This atlas holds radiomics features for normal pancreas and adrenal glands. If you need the underlying
voxel-level CT annotations, they are available in the public
[FLARE-MedFM/PancancerCTSeg](https://huggingface.co/datasets/FLARE-MedFM/PancancerCTSeg) collection
(MICCAI 2026 Challenge: Pan-cancer segmentation in CT scans,
[codabench](https://www.codabench.org/competitions/7149/)). Access is gated: accept the conditions on
the dataset page and export `HF_TOKEN` before downloading.

| Organ | Subset | Cases | Labels | Link |
|-------|--------|-------|--------|------|
| Pancreas | `train_per_cancer_type/Dataset006_PancreaticCancer` | 1797 | tumor (`labelsTr`) + organ mask (`labelsTr_organ`, 0 = background, 1 = organ, 2 = tumor) | [open](https://huggingface.co/datasets/FLARE-MedFM/PancancerCTSeg/tree/main/train_per_cancer_type/Dataset006_PancreaticCancer) |
| Adrenal glands | `train_per_cancer_type/Dataset005_AdrenalCancer` | 52 | tumor only, adrenocortical carcinoma (`labelsTr`) | [open](https://huggingface.co/datasets/FLARE-MedFM/PancancerCTSeg/tree/main/train_per_cancer_type/Dataset005_AdrenalCancer) |

The helper script in this repository lists these subsets, counts their files without downloading them,
and downloads only the parts you need:

```bash
python tools/flare_organs.py list                 # links, case counts, sources, licences
python tools/flare_organs.py inspect --organ all  # file counts and sizes via the HF API
pip install -U huggingface_hub
export HF_TOKEN=...                               # token is read from the environment, never stored
python tools/flare_organs.py download --organ all --out data
```

`download` also writes `data/FLARE_SOURCES.md` with the source datasets, licences and citations
(PanTS, PANORAMA, MSD-Pancreas, PanTrack, Adrenal-ACC-Ki67-Seg). The collection itself is licensed
CC BY-NC 4.0, while the licences of the original datasets differ.

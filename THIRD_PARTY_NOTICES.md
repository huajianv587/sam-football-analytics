# Third-Party Software, Models, and Data Notices

PitchVision orchestrates third-party runtimes on the GPU worker. Their source trees,
checkpoints, and model caches are downloaded into remote scratch storage and are not
committed to this repository. Each component remains governed by its own terms.
This notice is attribution and engineering documentation, not legal advice.

## Inference and game-state components

| Component | Pinned source / artifact | Role | Upstream terms |
| --- | --- | --- | --- |
| [Segment Anything 2](https://github.com/facebookresearch/sam2) | commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`; Base+ SHA256 `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5`; Large SHA256 `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318` | Base+ all-person video segmentation and optional selected-player Large refinement | [Apache License 2.0](https://github.com/facebookresearch/sam2/blob/main/LICENSE); the upstream repository documents additional notices for optional third-party code and demo fonts |
| [SoccerMaster](https://github.com/haolinyang-hlyang/SoccerMaster) | commit `2e5619712d93f634b841aaf37231cd9fceb6b262` | Reproducible packaging/reference for the football detector, TrackLab, SoccerNet GSR and weights | No root license file was present at the pinned revision. Do not redistribute or use commercially without confirming permission with its authors and the licenses of every embedded component |
| [SoccerNet Game State Reconstruction](https://github.com/SoccerNet/sn-gamestate) | Runtime copy supplied by the pinned SoccerMaster tree | Football detection, tracking and calibration pipeline | [GNU GPL v3](https://github.com/SoccerNet/sn-gamestate/blob/main/LICENSE) |
| [TrackLab](https://github.com/TrackingLaboratory/tracklab) | Runtime copy supplied by the pinned SoccerMaster tree | Offline multi-object tracking engine | [MIT License](https://github.com/TrackingLaboratory/tracklab/blob/main/LICENSE) |
| [PRTReID](https://github.com/VlSomers/prtreid) | commit `30617a75967e84d5d516959c4b84cbeea6f56493`; SoccerNet checkpoint MD5 `9633825232bc89f23a94522c5561650e` | Installed only for legacy profile compatibility; not used by the default field-space tracker | [Hippocratic License 3.0](https://github.com/VlSomers/prtreid/blob/master/LICENSE), including its use-based restrictions |
| [Ultralytics](https://github.com/ultralytics/ultralytics) / YOLOv8 | Football weight `yolo_v8x6_finetuned.pt`, SHA256 `c85259af82ae919294e9ec87457d73646682159a5511ec1df033fed84b15d03d` | Football-person bounding-box detection | [AGPL v3](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) or an applicable Ultralytics enterprise license; model-weight terms must also be checked before distribution or commercial use |
| [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO11 Segment | package `8.4.138`; COCO checkpoint `yolo11s-seg.pt`, SHA256 `1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3` | Sport-agnostic real-time `person` boxes and lightweight instance Masks | [AGPL v3](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) or an applicable Ultralytics enterprise license; model-weight terms must also be checked before distribution or commercial use |
| PnLCalib model weights | Zenodo artifacts `pnl_SV_kp` MD5 `322d4a6c82d2966ea88b69963ba85f07` and `pnl_SV_lines` MD5 `270b94527c9e817bc32edd54c8e47b62` | Per-frame pitch calibration | Distributed through the SoccerNet/SoccerMaster runtime; consult the corresponding upstream repository and Zenodo record before redistribution |
| [EasyOCR](https://github.com/JaidedAI/EasyOCR) | version `1.7.2` | Multi-frame jersey-number OCR | [Apache License 2.0](https://github.com/JaidedAI/EasyOCR/blob/master/LICENSE) |
| [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | optional revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`; not installed or run by the default fast profile | Optional role classifier for experiments | Governed by the model card and repository license at the pinned revision |

`backend/scripts/bootstrap_remote.sh` is the executable source of truth for
commits, revisions, URLs and checksums. It applies three small compatibility
patches to the remote runtime: removes eager imports for unused tracker wrappers,
uses positional pandas indexing in the PnLCalib batch, and fixes RGB/BGR input at
the detector boundary. No patched upstream source is committed here.

The repository's two-stage association and field-space scoring are original
integration code informed by the [ByteTrack paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf)
and [FieldMOT paper](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Chen_FieldMOT_A_Field-Registered_Multi-Object_Tracking_for_Sports_Videos_CVPRW_2025_paper.pdf).
No source code from either paper implementation is copied into this repository.

## General runtime dependencies

PyTorch, TorchVision, OpenCV, NumPy, SciPy, scikit-learn, FFmpeg, FastAPI,
Next.js, React, Supabase clients and their transitive dependencies are installed
from their official package channels. Their package distributions contain the
authoritative license texts. FFmpeg codec availability and licensing depend on
the cluster build.

## Football rosters and footage

The seed migration contains factual squad information transcribed from official
football-association and FIFA publications. Those publications are cited in the
migration and README; their page design, photographs and prose are not copied.
Match footage is not included. Users are responsible for rights to footage they
upload and for biometric, privacy and sports-data obligations in their jurisdiction.

## Distribution warning

The public PitchVision repository contains original orchestration, API, analytics
and UI code, but a working GPU runtime combines components with GPL, AGPL,
Hippocratic and potentially unspecified upstream terms. Review all upstream terms
before distributing a prebuilt runtime, offering a public service, or using the
system commercially.

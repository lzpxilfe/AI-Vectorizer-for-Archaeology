# ArchaeoTrace Academic & Technical References

Primary academic literature, computer vision architectures, geometric interpolation methods, geospatial engines, and independent external evaluations underlying **ArchaeoTrace** (`AI-Vectorizer-for-Archaeology`). All DOIs, arXiv identifiers, and URLs have been verified against primary publisher records.

## Label Legend (A / B / C)

- **(A)** External library or QGIS/GDAL/OpenCV/ONNX model invoked by ArchaeoTrace
- **(B)** Formula, geometric bridge, evidence pipeline, or safety policy implemented directly in `ai_vectorizer/core/`
- **(C)** Conceptual background, benchmark context, or independent external case study

---

## 1. Multi-Scale Ink Centerline & Morphological Line Evidence (`core/edge_detector.py`, `core/line_evidence.py`, `core/ink_color.py`)

**(B) Multi-scale Black Top-Hat morphological ink isolation (9 / 15 / 31 source pixels):**
> Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press, London. ISBN: 978-0126372403

**(A / B) Adaptive Otsu thresholding & 1-pixel morphological skeletonization (optional `scikit-image` / NumPy fallback):**
> Otsu, N. (1979). "A Threshold Selection Method from Gray-Level Histograms." *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), pp. 62–66. DOI: 10.1109/TSMC.1979.4310076
>
> Zhang, T.Y., & Suen, C.Y. (1984). "A Fast Parallel Algorithm for Thinning Digital Patterns." *Communications of the ACM*, 27(3), pp. 236–239. DOI: 10.1145/357994.358023

---

## 2. Bounded Direction-Aware Live-Wire & Shortest-Path Tracing (`core/livewire.py`, `core/path_finder.py`, `core/trace_guidance.py`)

**(B) Interactive Live-Wire / Intelligent Scissors boundary tracing & shortest-path search:**
> Mortensen, E.N., & Barrett, W.A. (1995). "Intelligent Scissors for Image Composition." *Proceedings of the 22nd Annual Conference on Computer Graphics and Interactive Techniques (SIGGRAPH '95)*, pp. 191–198. DOI: 10.1145/218380.218442
>
> Dijkstra, E.W. (1959). "A note on two problems in connexion with graphs." *Numerische Mathematik*, 1(1), pp. 269–271. DOI: 10.1007/BF01386390
>
> Hart, P.E., Nilsson, N.J., & Raphael, B. (1968). "A Formal Basis for the Heuristic Determination of Minimum Cost Paths." *IEEE Transactions on Systems Science and Cybernetics*, 4(2), pp. 100–107. DOI: 10.1109/TSSC.1968.300136

Note: ArchaeoTrace computes a bounded direction-aware shortest-path tree per accepted anchor so interactive mouse previews only walk predecessor indices, blends coordinates linearly from `0%` (literal cursor) to `100%` (full Live-Wire route), and supports explicit chromatic ink profiling and user-marked avoidance penalties.

---

## 3. Explicit Contour Gap Bridging (`core/manual_gap_bridge.py`, `core/confirmed_gap_curve.py`)

**(B) Parametric cubic Hermite curve interpolation from user-confirmed endpoints and local tangents:**
> Ferguson, J.C. (1964). "Multi-Variable Curve Interpolation." *Journal of the Association for Computing Machinery (JACM)*, 11(2), pp. 221–228. DOI: 10.1145/321217.321225

Note: Gap bridging (`G`) operates purely geometrically on two user-confirmed endpoints and bounded local tangents to preview a single smooth curve across printed labels or map breaks without invoking neural generation or auto-committing geometry.

---

## 4. Optional Local Deep Learning Guidance (Default OFF)

### 4.1 Smart Recovery — EfficientSAM-Ti Corridor Challenger (`core/efficientsam_onnx.py`, `core/smart_recovery.py`)

**(A) Pretrained EfficientSAM-Ti segmentation backbone (ONNX Runtime, fixed-hash local inference):**
> Xiong, Y., Varadarajan, B., Wu, L., Xiang, X., Xiao, F., Zhu, C., Dai, X., Wang, D., Sun, F., Iandola, F., Krishnamoorthi, R., & Chandra, V. (2024). "EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 16111–16121. DOI: 10.1109/CVPR52733.2024.01525

Note: Raw SAM masks are never saved directly as contour geometry or binary-OR'd with Ink. EfficientSAM-Ti serves strictly as a soft corridor challenger on low-confidence Ink segments and fails closed to the Ink route unless endpoint, detour, strong-Ink retention, and branch-switch safety checks pass.

### 4.2 Learned Contour Guidance (`core/learned_model.py`, `core/learned_prior.py`, `learning/model.py`)

**(A / B) ResNet-18 dual-head U-Net (`resnet18-smp-unet-small-dual-head-v1`, predicting `contour_support` and `other_ink_risk` via local ONNX Runtime):**
> Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." *Medical Image Computing and Computer-Assisted Intervention (MICCAI 2015)*, LNCS 9351, pp. 234–241. DOI: 10.1007/978-3-319-24574-4_28
>
> He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770–778. DOI: 10.1109/CVPR.2016.90

---

## 5. Advanced / Legacy Edge & Segmentation Backends (`core/edge_detector.py`, `core/sam_engine.py`)

**(A) LSD (Line Segment Detector, OpenCV):**
> Grompone von Gioi, R., Jakubowicz, J., Morel, J.-M., & Randall, G. (2010). "LSD: A Fast Line Segment Detector with a False Detection Control." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 32(4), pp. 722–732. DOI: 10.1109/TPAMI.2008.300

**(A) HED (Holistically-Nested Edge Detection, OpenCV DNN / Caffe):**
> Xie, S., & Tu, Z. (2015). "Holistically-Nested Edge Detection." *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pp. 1395–1403. DOI: 10.1109/ICCV.2015.164
>
> Extended journal version: Xie, S., & Tu, Z. (2017). "Holistically-Nested Edge Detection." *International Journal of Computer Vision*, 125(1–3), pp. 3–18. DOI: 10.1007/s11263-017-1004-z

**(A) MobileSAM:**
> Zhang, C., Han, D., Qiao, Y., Kim, J.U., Bae, S.-H., Lee, S., & Hong, C.S. (2023). "Faster Segment Anything: Towards Lightweight SAM for Mobile Applications." *arXiv preprint* arXiv:2306.14289. https://arxiv.org/abs/2306.14289

**(A) SAM (Segment Anything Model, ViT-B):**
> Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.-Y., Dollár, P., & Girshick, R. (2023). "Segment Anything." *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 4015–4026. DOI: 10.1109/ICCV51070.2023.00371

**(A) Legacy Canny Edge Detector:**
> Canny, J. (1986). "A Computational Approach to Edge Detection." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, PAMI-8(6), pp. 679–698. DOI: 10.1109/TPAMI.1986.4767851

---

## 6. Experimental DEM & Hillshade Reconstruction (`core/dem_pipeline.py`)

**(A) Linear TIN interpolation (`qgis:tininterpolation`) & GDAL Horn hillshade (`gdal:hillshade`):**
> Delaunay, B. (1934). "Sur la sphère vide. À la mémoire de Georges Voronoï." *Bulletin de l'Académie des Sciences de l'URSS*, VII série, 1934(6), pp. 793–800.
>
> Horn, B.K.P. (1981). "Hill shading and the reflectance map." *Proceedings of the IEEE*, 69(1), pp. 14–47. DOI: 10.1109/PROC.1981.11918
>
> GDAL/OGR contributors (2026). *GDAL/OGR Geospatial Data Abstraction software Library*. Open Source Geospatial Foundation. DOI: 10.5281/zenodo.5884351

---

## 7. Independent External Evaluation & Press

**(C) Independent century-scale landscape-change case study (1929–39 WIG & 1960s–80s Polish topographic maps vs. modern LiDAR DEM):**
> Walczak, S. (2026, October 24). "Digitising Historical Maps: Testing AI-Assisted Contour Tracing in QGIS." *Geoawesome*. https://geoawesome.com/digitising-historical-maps-testing-ai-assisted-contour-tracing-in-qgis/

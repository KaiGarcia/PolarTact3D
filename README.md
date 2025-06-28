# PolarTact3D

## Single-shot Tactile 3-D Shape & Color Sensing with Polarization Imaging

[Robotics: Science and Systems (RSS) 2025](https://roboticsconference.org/) - [Workshop on Navigating Contact Dynamics in Robotics](https://sites.google.com/colorado.edu/ws-contact-dynamics/home), 
Los Angeles, USA · 25 June 2025.

**🏆Best Paper Award** (3rd Prize)🏆

[**Paper (PDF)**](https://arxiv.org/abs/xxxxx) | [**Project Page**](https://polartact3d.github.io) | **Contact:** Kai Garcia <kaigarciadev@gmail.com>, Huaijin (George) Chen <huaijin@hawaii.edu>

---

## Abstract

Vision-based tactile sensors are essential for robotic applications such as grasping and physical interaction.  
We propose a **low-cost, polarization-imaging tactile sensor** that captures **both shape and color in a single shot**.  

Unlike photometric-stereo solutions (e.g., [GelSight](http://gelsight.com/)) that require precise internal illumination and reflective coatings—which block color capture—our method leverages the **Angle of Linear Polarization (AoLP)** and **Degree of Linear Polarization (DoLP)** to encode 3-D geometry. This enables robust shape reconstruction even on transparent or specular targets. 

The sensor is built from commercial transparent polyethylene (PE) film and an off-the-shelf [polarization camera](https://thinklucid.com/product/phoenix-5-mp-imx264/), making it simple and inexpensive to assemble. We validate the design with real-world experiments across diverse contact surfaces.

---

## Repository Layout

```text
PolarTact3D/
├── SfPUEL/               # Learning-based pipeline
│   ├── tools/       
│   │   └── visualize_results.ipynb # Visualize paper figures
├── physics_based/        # Physics-based Algorithms
│   ├── notebooks/
│   │   └── viz.ipynb     # Generates all paper figures
│   └── requirements.txt
└── README.md
```
#### You can find the .STEP file in /cad
---

## Quick Start

### 1 · Clone

```bash
git clone https://github.com/KaiGarcia/PolarTact3D.git
cd PolarTact3D
```

### 2 · Physics-based demo

```bash
cd physics_based
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/viz.ipynb          # generates Figures 3–6
```

### 3 · Learning-based SfPUEL demo

```bash
cd SfPUEL/tools
jupyter lab visualize_results.ipynb      # loads pretrained weights & runs on samples
```

---

## Citation
If you find this work useful for your research, please cite:
```bibtex
@inproceedings{garcia2025polartact3d,
  title     = {{PolarTact3D}: Single-shot Tactile 3-D Shape and Color Sensing with Polarization Imaging},
  author    = {Kai Garcia and Mairi Yoshioka and Huaijin Chen and Tyler Ray and Tianlu Wang and Frances Zhu},
  booktitle = {Robotics: Science and Systems Workshop on Navigating Contact Dynamics in Robotics},
  year      = {2025},
  address   = {Los Angeles, USA},
  url       = {https://arxiv.org/abs/xxxxx},
  note      = {Workshop paper at RSS 2025}
}
```
The shape-from-polarization network used in this work is based on:
```bibtex
@inproceedings{lyu2024sfpuel,
  title     = {Sf{PUEL}: Shape from Polarization under Unknown Environment Light},
  author    = {Youwei Lyu and Heng Guo and Kailong Zhang and Si Li and Boxin Shi},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024}
}
```

---

## License

Released under the **MIT License**. See `LICENSE` for details.

---

## Ackonwledgement

The presentation of the work at the RSS 2025 is made possible through the NSF AI Institutes Virtual Organization (AIVO) Travel Grant Program. Thank you, NSF! The $250 prize money is sponsored by Apple Inc. Thank you, organizers and Apple!

---

*Enjoy PolarTact3D!*

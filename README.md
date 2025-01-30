# Membrane Analysis Toolbox
With this toolbox you can analyse **molecular dynamics (MD)** simulation data to investigate properties of **nanoporous carbon materials (NCMs)**.  
It has been developed in the group of Kristyna Pluhackova at the University of Stuttgart for analysing **NCM** and their properties such as diffusion. 

## Main Components:
- TransitionPathAnalysis class with the following key features:
    - Analyse passages through membrane and their **passage time distribution**
    - Calculate the **diffusion coefficient** for a solvent using the first passage time theory (FPT)
- EffectivePoreSizeAnalysis class with the following key features:
    - Analyse the **effective pore radius** of a membrane pore
    - Analyse the density of molecules in the pore using kernel methods for a **density heatmap**

## Installation:
AnalysisForNanoporousCarbonMaterials is developed using python 3.12 and installing it requires a minimum of python 3.12.

1) Download or clone this repository
2) Run `pip install [path to this package]` in your python environment

## Usage
- Look at `examples/` to see how the package can be used if you want to write your own code. 
- Find the code and how the package was used for the paper in `usage/`

## Acknowledgement
This Toolbox uses work by Gotthold Fläschner.

## Citing
When you want to use this work, cite the Article.

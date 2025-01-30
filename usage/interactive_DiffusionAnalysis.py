import os
import time

import matplotlib.pyplot as plt

from AnalysisForNanoporousCarbonMaterials.DiffusionAnalysis import DiffusionAnalysis
from AnalysisForNanoporousCarbonMaterials.MembraneStructures import (
    CubicMembrane,
    HexagonalMembrane,
    Solvent,
)

print("\n\nInteractive Analysis of Diffusion in Membranes")
print("===============================================")
print("\nFirst enter the paths to the simulation files.")
print("Remember to end the path with a '/'.")

no_valid_topol_file = True
while no_valid_topol_file:
    print("\nEnter the path to the topol.tpr file: ")
    topol_path = input("-> ")
    print("\nEnter Filename (press ENTER if it is topol.tpr): ")
    topol_file_name = input("-> ")
    if topol_file_name == "":
        topol_file_name = "topol.tpr"
    topol_file = topol_path + topol_file_name
    if os.access(topol_file, os.R_OK):  # check if file exists and is readable
        no_valid_topol_file = False
    else:
        print("File not found or missing read permission. Please try again.")

no_valid_traj_file = True
while no_valid_traj_file:
    print(
        "\nEnter the path to the traj.xtc file (press ENTER if it is in the same path as topol.tpr): "
    )
    traj_path = input("-> ")
    if traj_path == "":
        traj_path = topol_path
    print("\nEnter Filename (press ENTER if it is traj.xtc): ")
    traj_file_name = input("-> ")
    if traj_file_name == "":
        traj_file_name = "traj.xtc"
    traj_file = traj_path + traj_file_name
    if os.access(traj_file, os.R_OK):  # check if file exists and is readable
        no_valid_traj_file = False
    else:
        print("File not found or missing read permission. Please try again.")

print("\nEntere the type of membrane in the Simulation. Possible types are:")
print("'C': Cubic Membrane \n'H': Hexagonal Membrane \n'S': Solvent")
no_valid_membrane_type = True
while no_valid_membrane_type:
    membrane_type = input("-> ")
    if membrane_type in ["C", "H", "S"]:
        no_valid_membrane_type = False
    else:
        print("Invalid input. Please try again.")


analysis_max_step_size_ps = None
match membrane_type:
    case "C":
        structure = CubicMembrane(
            selectors="resname C",
            cube_arrangement=(2, 2, 2),
            cube_size=90,
            pore_radius=15,
        )
        analysis_max_step_size_ps = 200  # use 200 since it was used in the analyses from the beginning and it is a good vale. We do not expect a transition to be faster than that.
    case "H":
        print(
            "\nEnter the length of the membrane in Angstrom or press ENTER to select the default of 180A: "
        )
        L = input("-> ")
        L = 180 if L == "" else int(L)
        structure = HexagonalMembrane(
            selectors="resname C",
            L=L,
        )
        analysis_max_step_size_ps = 200  # use 200 since it was used in the analyses from the beginning and it is a good vale. We do not expect a transition to be faster than that.
    case "S":
        print("\nEnter the lower Z value (in Angstrom): ")
        lowerZ = int(input("-> "))
        print("\nEnter the upper Z value (in Angstrom): ")
        upperZ = int(input("-> "))
        L = upperZ - lowerZ
        structure = Solvent(
            lowerZ=lowerZ,
            upperZ=upperZ,
            L=L,
        )
        analysis_max_step_size_ps = (
            2  # use 2 because the transitions in the solvent case are much faster
        )
        # TODO make this an interactive input
    case _:
        raise ValueError("Invalid input for membrane_type")

# Check if the user wants to save the results and if so, where
results_dir = None
print("\nDo you want to save the results? ('y' or press ENTER for no)")
want_to_save_results = input("-> ") == "y"

if want_to_save_results:
    print(
        "\nSave results path (has to end with '/'; press ENTER to skip and save to standard /analyis folder in Simulation folder): "
    )
    results_dir_input = input("-> ")
    if results_dir_input == "":
        results_dir = traj_path + "analysis/"
    else:
        results_dir = results_dir_input

# STEP 1: initialise the Data into the class
DA = DiffusionAnalysis(
    topology_file=topol_file,
    trajectory_file=traj_file,
    results_dir=results_dir,
    analysis_max_step_size_ps=analysis_max_step_size_ps,
    verbose=True,
    membrane=structure,
)

print(DA)

if isinstance(DA.membrane, Solvent):
    DA.print_membrane_location()
else:
    # measure start time to measure time
    start = time.time()
    DA.find_membrane_location()
    print(f"Time to find membrane location: {int(time.time() - start)}s")
    DA.print_membrane_location()
    fig_ml = DA.verify_membrane_location()
    if want_to_save_results:
        DA.save_fig_to_results(fig=fig_ml, name="membrane_location_verification")
plt.show()

wants_to_analyse = True
print("Analyse the transitions of atoms here:")
while wants_to_analyse:
    resname = input("Enter the resname (example: HEX): ")
    name = input("Enter the name (example: C1): ")
    selector = f"resname {resname} and name {name}"
    short = resname.lower() + "_" + name.lower()

    # perform the analysis (former method analyse_resname())
    print(f"\n{short} analysis")

    start = time.time()
    DA.calc_passagetimes(selector)
    print(f"\t{short}-passages: " + str(len(DA.passageTimes[selector])))
    print(f"Time to calculate passage times: {int(time.time() - start)}s")
    # DA.plot_passagetimedist(selector)

    if want_to_save_results:
        DA.save_passage_times_in_ns_to_txt(selector, short + "_passagetimes_in_ns.txt")

    print(
        "For the calculation of the Diffusion coefficient, an initial guess has to be made."
    )
    print(
        f"The suggested guess (by approximation of the mean) is {DA.guess_D(selector)}"
    )
    print(
        "It is important to check the plot of the passage time distribution and the fit to see if the global minimum of the fit was found."
    )
    D_guess = input(
        f"Enter the guess for the diffusion coefficient (ENTER for using the suggested guess ({DA.guess_D(selector)})): "
    )
    if D_guess == "":
        D_guess = DA.guess_D(selector)
    else:
        if "," in D_guess:
            D_guess = float(D_guess.replace(",", "."))
        else:
            D_guess = float(D_guess)
    DA.calc_diffusion(selector, D_guess)
    print(f"\t{short}-Diffusioncoefficient: " + str(DA.D[selector]).replace(".", ","))
    fig_PDF = DA.plot_diffusion(selector)
    if want_to_save_results:
        DA.save_fig_to_results(fig=fig_PDF, name="diffusion_PDF_" + short)

    fig_sp = DA.plot_starting_points(selector)
    if want_to_save_results:
        DA.save_fig_to_results(fig=fig_sp, name="starting_points_" + short)

    plt.show()

    wants_to_analyse = (
        input(
            "\n\n\n Do you want to analyse the transitions of another resname? (y/n) "
        )
        == "y"
    )

if want_to_save_results:
    DA.store_results_json()
print(DA)
plt.show()
print("\n\n\n\n\n")

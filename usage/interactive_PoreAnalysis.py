import matplotlib.pyplot as plt
from AnalysisForNanoporousCarbonMaterials.MembraneStructures import (
    CubicMembrane,
    HexagonalMembrane,
    Solvent,
)
from AnalysisForNanoporousCarbonMaterials.PoreAnalysis import PoreAnalysis

print("\n\nInteractive Analysis of the Membrane Pore")
print("===============================================")
print("\nFirst enter the paths to the simulation files.")
print("Remember to end the path with a '/'.")
print("\nEnter the path to the topol.tpr file: ")
topol_path = input("-> ")
print("\nEnter Filename (press ENTER if it is topol.tpr): ")
topol_file_name = input("-> ")
if topol_file_name == "":
    topol_file_name = "topol.tpr"
topol_file = topol_path + topol_file_name
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

print("\nEntere the type of membrane in the Simulation. Possible types are:")
print("'C': Cubic Membrane \n'H': Hexagonal Membrane \n'S': Solvent")
membrane_type = input("-> ")

match membrane_type:
    case "C":
        structure = CubicMembrane(
            selectors="resname C",
            cube_arrangement=(2, 2, 2),
            cube_size=90,
            pore_radius=15,
        )
    case "H":
        print(
            "\nEnter the length of the membrane in Angstrom or press ENTER to select the default of 180: "
        )
        L = input("-> ")
        L = 180 if L == "" else int(L)
        # ask for selectors of membrane
        print(
            "\nEnter the selectors for the membrane atoms. Example: 'resname C' or 'resname C, resname CO'"
        )
        selectors = input("-> ")
        selectors = selectors.split(", ")
        structure = HexagonalMembrane(
            selectors=selectors,
            L=L,
        )
    case "S":
        print("\nEnter the lower Z value: ")
        lowerZ = int(input("-> "))
        print("\nEnter the upper Z value: ")
        upperZ = int(input("-> "))
        L = upperZ - lowerZ
        structure = Solvent(
            lowerZ=lowerZ,
            upperZ=upperZ,
            L=L,
        )
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


PA = PoreAnalysis(
    topology_file=topol_file,
    trajectory_file=traj_file,
    membrane=structure,
    analysis_max_step_size_ps=200,  # use 200 since it was used in the analyses from the beginning and it is a good vale. We do not expect a transition to be faster than that.
    results_dir=results_dir,
    verbose=True,
)

print(PA)
if isinstance(PA.membrane, Solvent):
    PA.print_membrane_location()
else:
    PA.find_membrane_location()
    PA.print_membrane_location()
    fig_ml = PA.verify_membrane_location()
    if want_to_save_results:
        PA.save_fig_to_results(fig=fig_ml, name="membrane_location_verification")
plt.show()

wants_to_analyse = True
while wants_to_analyse:
    print("\nWhat do you want to analyse?")
    print("\tA: Effective Pore Radius")
    print("\tB: Density (KDE) of the Solvent & Membrane normed")
    print("\tC: Density (KDE) of the Solvent")
    print("\tQ: Quit")

    analysis_type = input("-> ")
    match analysis_type:
        case "A":
            print(
                "plot x, y, z histograms to validate the z-constraints and to find the y-constraints"
            )
            fig_ac = PA.analyseConstraints(
                "resname C",  # TODO: no hardcoded values
                y_constraints=(
                    25,
                    35,
                ),  # TODO use find pore to suggest values for the plot
                z_constraints=PA.membrane.find_zConstraints(),
            )
            if want_to_save_results:
                PA.save_fig_to_results(fig=fig_ac, name="analyse_constraints_xyz")
            plt.show()
            print(
                "\nWhat y_constraints do you want to use for the analysis (in Angstrom)? Example '25, 35'. This means that the analysis will be done between 25 and 35 Angstrom in the y direction."
            )
            y_constraints = input("-> ")
            y_constraints = y_constraints.split(", ")
            # ask for the solvent selectors
            print(
                "\nEnter the selectors for the solvent atoms. Example: 'resname HEX and name C1' or 'resname HEX and name C1, resname DOD and name C2'"
            )
            solvent_selectors = input("-> ")
            solvent_selectors = solvent_selectors.split(", ")
            edges, fig_eps = PA.calculateEffectivePoreSize(
                solvent_selectors=solvent_selectors,
                z_constraints=PA.membrane.find_zConstraints(),
                y_constraints=(int(y_constraints[0]), int(y_constraints[1])),
                strategy="intersection",
                bins=50,
            )
            if want_to_save_results:
                PA.save_fig_to_results(fig=fig_eps, name="effective_pore_size")
            plt.show()
            print(edges)
            print(f"The found effective pore edges are: {edges} in Angstrom")
            print(f"The effective pore size is: {edges[1] - edges[0]} Angstrom")
        case "B":
            print(
                "\nWhich atoms would you like to consider for the normed density plot? Example: 'resname DOD and name C3' or 'resname C, resname DOD and name C2'"
            )
            atoms = input("-> ")
            selectors = atoms.split(", ")
            _, fig_dn = PA.analyseDensityNormed(
                selectors=selectors,
                z_range=PA.membrane.find_zConstraints(),
                skip=1000,
                bw=0.13,
            )
            if want_to_save_results:
                PA.save_fig_to_results(
                    fig=fig_dn, name="density_normed_" + str(selectors)
                )
            plt.show()
        case "C":
            print(
                "\nWhich atoms would you like to consider for the density plot? Example: 'resname DOD and name C3' or 'resname HEX and name C1, resname DOD and name C2'"
            )
            atoms = input("-> ")
            selectors = atoms.split(", ")
            _, fig_d = PA.analyseDensity(
                selectors=selectors,
                z_range=PA.membrane.find_zConstraints(),
                skip=10,
                bw=0.13,
            )
            if want_to_save_results:
                PA.save_fig_to_results(fig=fig_d, name="density_" + str(selectors))
            plt.show()
        case "Q":
            break
        case _:
            print("Invalid input, try again.")
            wants_to_analyse = (
                input(
                    "\nDo you want to analyse something else? ('y' or press ENTER for no)"
                )
                == "y"
            )

    wants_to_analyse = (
        input("\nDo you want to analyse something else? ('y' or press ENTER for no)")
        == "y"
    )

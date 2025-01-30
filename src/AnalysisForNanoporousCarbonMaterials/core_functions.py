import numpy as np
from scipy.optimize import basinhopping, curve_fit, dual_annealing, least_squares
from statsmodels.distributions.empirical_distribution import ECDF

"""
This is a collection of core functions that are used by the analysis classes,
but can also be used standalone as imported functions
"""


def findPassages(T, isAtomAbove, isAtomBelow, p=1, p_middle=1) -> tuple:
    """
    Measure start and endpoint of passages through the bounds.
    Returns the array of starting times and the array of endtimes (not in ns, but in timesteps!)

    Args:
        T (np.ndarray): trajectories: 3D array with shape (number of trajectories, number of timesteps, 3)
        isAtomAbove (function): function that returns True if the atom is above the membrane
        isAtomBelow (function): function that returns True if the atom is below the membrane
        p (int, optional): timesteps, that an object has to be above or below a
        bound to be seen as above or below. different values than 1 can make sense
        to compensate uncontinuous behavior (see documentation for more details). Defaults to 1.
        p_middle (int, optional): timesteps, that an object has to be in the middle to
        be seen as passaging through the middel. 3 means, the object has to be in
        the middle for 3 timesteps to be counted as a valid transition. Defaults to 1 because it is the
        most basic definition of a transition.

    Returns:
        _type_: flip start und end times (in timesteps) in an array and the indizes of the S file which
        trajectories have a transition; ffs is the last timestep where the traj is outside
        the bounds and ffe is the first timestep where the traj is outside the bounds again

    Raises:
        Exception: no transition detected. Check traj files and boundaries
        AttributeError: 'list' object has no attribute 'astype' if the list of starting times ffs/ffe/indizes is empty and no passage was detected; check boundaries and trajectories

    """
    # TODO avoid using for loops and instead use numpy methods
    number_of_traj = T[:, 0, 0].size
    number_of_timesteps = T[0, :, 0].size
    label = np.zeros(number_of_traj)  # how is the object labeled
    middle_count = np.zeros(number_of_traj)  # how long in the middle
    lower_count = np.zeros(number_of_traj)  # how long in the lower layer
    upper_count = np.zeros(number_of_traj)  # how long in the upper layer
    full_flips_start = []
    full_flips_end = []
    indizes = []
    for t in range(number_of_timesteps):
        for a in range(number_of_traj):
            curr = T[a, t]
            # print(t, curr, label[a], label_count[a], middle_count[a])
            if isAtomBelow(curr):  # object is below lower bound
                lower_count[a] = lower_count[a] + 1  # one time step longer in the layer
                upper_count[a] = 0  # set count of upper layer to 0
                if lower_count[a] == p:
                    if label[a] == 4:  # if it comes from above
                        full_flips_start = np.append(
                            full_flips_start, t - middle_count[a] - p
                        )
                        # end is the current timestep -p, beccause it alredy has been in the layer p steps before;
                        full_flips_end = np.append(full_flips_end, t - p + 1)
                        indizes = np.append(indizes, a)
                    label[a] = 1  # label, that its now in the lower layer
                    # set middle count to 0 (only in this if branch, because middle count
                    # (time count) should
                    # go on if the object only slips out the middle for less than p timesteps)
                    middle_count[a] = 0

            elif isAtomAbove(curr):
                upper_count[a] = upper_count[a] + 1
                lower_count[a] = 0
                if upper_count[a] == p:
                    if label[a] == 2:
                        full_flips_start = np.append(
                            full_flips_start, t - middle_count[a] - p
                        )
                        full_flips_end = np.append(full_flips_end, t - p + 1)
                        indizes = np.append(indizes, a)
                    label[a] = 5
                    middle_count[a] = 0

            # if not (isAtomBelow(curr) or isAtomAbove(curr)):
            else:
                # one timestep longer in the middle
                middle_count[a] = middle_count[a] + 1
                lower_count[a] = 0
                upper_count[a] = 0
                if middle_count[a] == p_middle:
                    # if its ready to be counted as beeing in the middle
                    if label[a] == 1:
                        label[a] = 2  # label as coming from below
                    if label[a] == 5:
                        label[a] = 4  # label as coming from above

    if len(indizes) == 0:
        raise Exception("no transition detected. Check traj files and boundaries")

    # return the start and end times of all passages that meet the conditions
    # that are set by defining p and p_middle
    return full_flips_start.astype(int), full_flips_end.astype(int), indizes.astype(int)


def save_1darr_to_txt(arr: np.ndarray, path: str):
    """
    Save a NumPy array to a text file.

    Args:
        arr (numpy.ndarray): The array to be saved.
        path (str): The path to the output text file including the extension .txt.

    Returns:
        None
    """
    try:
        with open(path, "w") as f:
            for i in range(0, arr.size):
                f.write("\n")
                f.write(str(arr[i]))
    except PermissionError:
        print(
            f"PermissionError: Permission denied to {path}. The results will not be saved."
        )


def fit_diffusion_pdf(L: float, passage_times: list, D_guess: float) -> float:
    """
    calculate diffusion using Gotthold Fläschner Script and a PDF fit.

    Args:
        L: length of the membrane in Angstrom
        passage_times: passage times in ns
        D_guess: initial guess for the diffusion coefficient

    Returns:
        D_hom: diffusion coefficient calculated using PDF fit in Angstrom^2/ns
    """
    print(f"Calculating diffusion coefficient using a PDF fit ...")
    ecdf = ECDF(passage_times)

    # PREPARE DATA
    idx = (np.abs(ecdf.y - 0.5)).argmin()
    centertime = ecdf.x[idx]
    bins = int(10 * np.max(passage_times) / centertime)
    histo, edges = np.histogram(passage_times, bins, density=True)
    center = edges - (edges[2] - edges[1])
    center = np.delete(center, 0)
    edges = np.delete(edges, 0)

    params_hom = fitting_hom_lsq(center[0:], histo[0:], L, D_guess)
    D_hom = params_hom[0]
    return D_hom


#########################################################################################
# START Functions from Gotthold Fläschners script #######################################
# These functions implement a PDF fit for fitting the first passage time approach
# to the distribution of first passage times of molecules through a membrane.
# The first passage time approach is described in the paper by van Hijkoop et al.
# (https://doi.org/10.1063/1.2761897)
def hom(x, D, i, L):
    t = (L) ** 2 / (i**2 * np.pi**2 * D)
    return (-1) ** (i - 1) * i**2 * np.exp(-x / t)  # summand in equation 9 vanHijkoop


def fitfunc_hom(x, D, L):
    i = 151  # not to infinity but to 151 (approximation of the sum)
    result = 0
    for j in range(1, i):
        result = result + hom(x, D, j, L)
    return 2 * np.pi**2 * D / (L) ** 2 * result  # sum of equation 9 vanHijkoop


def fitfunc_hom_lsq(L):
    def f(D, x, y):
        n = 151
        sum = 0
        # print("start sum")
        for j in range(1, n):
            sum = sum + hom(x, D, j, L)
            # print(np.sum(hom(x, D, j, L)))
        eq = 2 * np.pi**2 * D / (L) ** 2 * sum
        residuals = eq - y
        return residuals  # difference vector between the fit using D and the data y

    return f


def fitting_hom_lsq(x_data, y_data, L, D0):
    """
    least squares depends on an initial value for D which has to be provided
    in order to find the right global minimum. It has shown that the PDF fit
    errors have not just one global min but several local minima. Therefore
    the initial value is required as a user input for the least squares fit.
    basin hopping is not suitable since it also depends on the initial value
    and the global minimum is not found in all cases.
    dual annealing also did not work since the bound for D is infinity on the
    positive axis.
    """
    res_robust = least_squares(
        fitfunc_hom_lsq(L), x0=D0, loss="soft_l1", args=(x_data, y_data), f_scale=0.3
    )
    return res_robust.x


# END Functions from Gotthold Fläschners script #########################################
#########################################################################################

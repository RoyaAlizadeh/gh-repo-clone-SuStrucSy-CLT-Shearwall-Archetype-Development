#25 November 2025
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
import seaborn as sns
from collections import defaultdict
import time
from functools import lru_cache
import json
def CLT_resistance_calculator(model_data, file_path='gh-repo-clone-SuStrucSy-CLT-Shearwall-Archetype-Development/CLT Properties.xlsx'):
    """
    Calculate the factored resistances of CLT wall panels using CSA-O86:24 guidelines
    and manufacturer-specific data (e.g., Nordic).

    Parameters:
    -----------
    model_data : dict
        Dictionary containing geometric and material information for the wall panel.
    file_path : str
        File path to the Excel file containing manufacturer CLT properties.

    Returns:
    --------
    tuple
        Contains dictionary of CLT properties and computed values:
        (CLT, Pr, Mr, Vr, Qr, PEV)
    """
    # Basic Units
    m = 1.0
    N = 1.0
    mm = m / 1000.0
    kN=1000*N
    Pa = N / (m ** 2)
    MPa = Pa * 1.0e6
    GPa = Pa * 1.0e9
    # Load CLT design properties from Excel
    clt_data = pd.read_excel(file_path, sheet_name=None)
    df = clt_data[list(clt_data.keys())[0]]  
    columns = df.columns[1:]

    # Parse the Excel data into a dictionary
    CLT = {
        column: {row[0]: row[columns.get_loc(column) + 1] for row in df.itertuples(index=False)}
        for column in columns
    }

    # Input parameters from model_data
    H = model_data['H']  # Wall height [mm]
    q = model_data['q_value']  # Applied lateral load [kN/m]
    bs = model_data['Panel_width']  # Single panel width [m]
    m_panel = model_data['m_panels']  # Number of panels
    t = model_data['t']  # Panel thickness [mm]
    layup = int(model_data['Layup'][0])  # Layup configuration
    n_perp_layers = int(layup/2)
    n_parl_layers = layup - n_perp_layers
    E_perp = 9000 * MPa
    E_parl = 11700 * MPa
    E_eq = (n_perp_layers* E_perp + n_parl_layers * E_parl)/layup


    B = bs * m_panel  # Total wall width [m]

    # Material properties for stress grade E1 (Table 8.2 CSA-O86)
    E05 = 0.82 * 11700 * MPa  # Effective modulus of elasticity [MPa]
    fc = 19.3 * MPa           # Compression parallel to grain [MPa]
    fcp = 5.3 * MPa           # Compression perpendicular to grain [MPa]

    # Modification factors from CSA-O86
    KD = Ks = KT = KH = 1  # Standard load duration, service condition, treatment, and system factors

    # --- Compressive Resistance Parallel to Grain (CSA 8.1.4.5.5) ---
    phiP = 0.8  # Resistance factor for axial compression
    Ke = 1      # Effective length factor
    Le = Ke * H  # Effective column length [mm]
    reff = CLT[layup]['reff0'] * B * mm  # Effective radius of gyration [mm]
    Aeff = CLT[layup]['Aeff'] * B * 1e3 * mm**2  # Effective area [mm²]

    Cc = Le / (np.sqrt(12) * reff)  # Slenderness ratio
    Kzc = min(6.3 * np.sqrt(12) * reff * H/mm * (-0.13), 1.3)  # Size factor
    Fc = fc * KD * KH * Ks * KT  # Adjusted compressive strength
    Kc = (1 + Fc * Kzc * Cc**3 / (35 * E05 * Ks * KT))**(-1)  # Slenderness factor

    Pr = phiP * Fc * Aeff * Kzc * Kc  # Factored axial compressive resistance [N]

    # --- Bending Resistance (CSA 8.1.4.3.1) ---
    Krb = 0.85  # Adjustment factor
    phiM = 0.9  # Resistance factor for bending
    fb = 28.2 * MPa

    I_eff_e = t*bs**3 /12
    EI_eff_e = E_eq * I_eff_e
    S_eff = EI_eff_e/E_parl *2/H
    # Mr = CLT[layup]['Mr0'] * B * kN * m * Krb * phiM  # Factored moment resistance [kN·m]
    KH = 1
    Mr_e = phiM * fb * KH * Ks * KT * KD * S_eff * Krb
    # EDGE-WISE BENDING
    #EDGE-WISE (EI)eff


    # --- Shear Resistance (CSA 8.1.4.4) ---
    phiV = 0.9  # Resistance factor for shear
    Vr_f = CLT[layup]['Vr0'] * B * kN * m * phiV  # Factored shear resistance [kN]
    Ag = bs * t
    fs = 0.5 * MPa
    Vr_e = phiV * fs *  KH * Ks * KT * KD * 2 * Ag /3

    # --- Compression Perpendicular to Grain (CSA 8.1.4.7.2) ---
    phiQ = 0.8  # Resistance factor
    KD = 1.15 #Short-term
    Fcp = fcp * KD * Ks * KT  # Adjusted compressive stress
    KB = Kzcp = 1  # Bearing length and size factors
    Qr = phiQ * Fcp * t * bs / 10 * KB  # Factored bearing resistance [N]

    # --- Euler Buckling Resistance ---
    GAeff = CLT[layup]['GAeff0'] * B * 1e6 * N  # Shear stiffness [N]
    Ieff = CLT[layup]['Ieff0'] * B * 1e6 * mm**4  # Effective moment of inertia [mm⁴]
    PE = (np.pi ** 2) * E05 * Ks * KT * Ieff / ((H / 2) ** 2)  # Euler buckling load in the plane of the applied moment
    PEV = PE / (1 + PE / GAeff)  #Euler buckling load in the plane of the applied bending moment adjusted for shear deformation

    return CLT, Pr, Mr_e, Vr_e, Qr, PEV

def Panel_capacity_check_I(Pr, Mr, PEV, Pf, t):
    """
    Checks compliance with CSA-O86:24 Clause 8.1.4.6 — Resistance to combined flatwise bending 
    and axial compressive load in CLT wall panels.

    Parameters:
    -----------
    Pr : float
        Factored compressive resistance under axial load [N]
    Mr : float
        Factored flatwise bending moment resistance [N·mm or kN·m]
    PEV : float
        Euler buckling load in-plane, adjusted for shear deformation [N]
    Pf : float
        Applied factored axial compressive load on the panel [N]
    t : float
        Panel thickness [mm]

    Returns:
    --------
    tuple
        (status, interaction_ratio)
        status = 1 if interaction ratio ≤ 1 (design is acceptable),
                 0 otherwise (design does not meet requirement)
        interaction_ratio = computed interaction ratio per Clause 8.1.4.6
    """
    # Basic Units
    m = 1.0
    N = 1.0
    mm = m / 1000.0
    kN=1000*N
    Pa = N / (m ** 2)
    MPa = Pa * 1.0e6
    GPa = Pa * 1.0e9

    # Factored bending moment due to axial load eccentricity (P-Delta effect)
    Mf = Pf * (t / 2)  # Assumes centroidal offset is half the thickness

    # Interaction check equation (CSA-O86:24, Clause 8.1.4.6)
    interaction_ratio = (Pf / Pr) + (Mf / Mr) * (1 / (1 - Pf / PEV))

    # Acceptable if interaction ratio is ≤ 1.0
    status = 1 if interaction_ratio <= 1.0 else 0

    return status, interaction_ratio

def calculate_induced_95th_force (kh,nf,kf,F,H,bs,m_panels,q,Beta):
    k_prime = kh+(m_panels-1)*nf*kf
    Teta = 1/(k_prime*Beta**2)(F*H/bs**2 - m_panels*q*(Beta-0.5))
    Fh = kh * Teta*bs * Beta
    Ff = nf*kf*Teta*bs*Beta
    Fq = q*bs
    return Fh,Ff,Fq

def Panel_capacity_check_II(Omega, Pf, Pg, Vf, Mf, Pr, Vr, Mr, Qr,Rc):
    """
    Checks panel design resistance for non-dissipative CLT elements per CSA-O86:24 Clause 11.9.3.6.4.1.

    This function verifies the factored resistance of CLT panels under combined
    seismic and gravity loads, ensuring all four limit states (axial, shear, moment, and bearing)
    satisfy the following inequality:
        F_r,ND ≥ α_T * F_j,ND + F_g,ND

    Parameters:
    -----------
    Omega : float
        Storey over-strength coefficient (α_T), min(Ω_i) from Clause 11.9.3.6.4.2
    Pf : float
        Axial force due to seismic load [N]
    Pg : float
        Axial force due to gravity load [N]
    Vf : float
        Shear force due to seismic load [N]
    Mf : float
        Moment due to seismic load [N·mm or kN·m]
    Pr : float
        Factored axial resistance [N]
    Vr : float
        Factored shear resistance [N]
    Mr : float
        Factored moment resistance [N·mm or kN·m]
    Qr : float
        Factored bearing resistance [N]

    Returns:
    --------
    int
        1 if all limit states pass, 0 otherwise. Prints a warning for each failed check.
    """
    # Basic Units
    m = 1.0
    N = 1.0
    mm = m / 1000.0
    kN=1000*N
    Pa = N / (m ** 2)
    MPa = Pa * 1.0e6
    GPa = Pa * 1.0e9

    # --- Axial Load Check ---
    P_total = Omega * Pf + Pg
    Axial_check = int(P_total <= Pr)
    # if not Axial_check:
    #     print("Panel Capacity Check II: Axial check failed")
    
    # --- Shear Load Check ---
    Shear_check = int(Omega * Vf <= Vr)
    # if not Shear_check:
    #     print("Panel Capacity Check II: Shear check failed")

    # --- Bearing Load Check ---
    Bearing_check = int(Rc <= Qr)
    # if not Bearing_check:
    #     print("Panel Capacity Check II: Bearing check failed")

     # --- Bending Moment Check ---
    Bending_check = int(Mf <= Mr)
    # if not Bearing_check:
    #     print("Panel Capacity Check II: Bearing check failed")   

    # Return 1 only if all checks pass
    return int(all([Axial_check, Shear_check,Bearing_check, Bending_check]))
    # return int(all([Axial_check]))

def rf_percentile(rf_mean, cov, percentile):
    """
    Calculates the specified percentile of a resistance factor based on a given mean and coefficient of variation (COV).

    Parameters:
    - rf_mean (float): Mean value of the resistance factor.
    - cov (float): Coefficient of Variation (COV), defined as the ratio of standard deviation to the mean.
    - percentile (float): Desired percentile to calculate (e.g., 15 for the 15th percentile).

    Returns:
    - float: Calculated resistance factor at the specified percentile.

    Steps:
    1. Compute the standard deviation from the mean and COV.
    2. Obtain the Z-score corresponding to the desired percentile.
    3. Calculate the resistance factor at the given percentile using the normal distribution assumption.
    """
    # Basic Units
    m = 1.0
    N = 1.0
    mm = m / 1000.0
    kN=1000*N
    Pa = N / (m ** 2)
    MPa = Pa * 1.0e6
    GPa = Pa * 1.0e9

    # Calculate standard deviation based on mean and COV
    rf_std = cov * rf_mean

    # Determine Z-score for the desired percentile
    z_score = stats.norm.ppf(percentile / 100)

    # Calculate resistance factor at the specified percentile
    rf_percentile_value = rf_mean + z_score * rf_std

    return rf_percentile_value

def WFT_check(kf, rf, nf, kh, rh, WB, m_panel, bs, H, q, Vf, Mf,COV):
    """
    Checks the adequacy of Wall-Floor Tension (WFT) connections in a CLT shearwall configuration.

    Parameters:
    - kf (float): Stiffness of one connection in a vertical joint (kN/m).
    - rf (float): Mean resistance of one connection in a vertical joint (kN).
    - nf (int): Number of connections per vertical joint.
    - kh (float): Hold-down stiffness in the uplift direction (kN/m).
    - rh (float): Factored hold-down resistance in the uplift direction (kN).
    - WB (str): Desired wall behavior ('CP' for coupled-panel or 'SW' for single-wall).
    - m_panel (int): Number of CLT panels in the shearwall.
    - bs (float): Length of one CLT panel within the shearwall (m).
    - H (float): Height of the shearwall (m).
    - q (float): Total factored dead load applied at the top of the shearwall (kN/m).
    - Vf (float): Factored lateral force applied at the top of the shearwall (kN).
    - COV (float): Coefficient of variation for the resistance factor.

    Returns:
    - int: Returns 1 if the connection meets the requirements, otherwise returns 0.

    Steps:
    1. Determine the kinematic mode of the CLT shearwall ('CP', 'SW').
    2. Compute limits for connection stiffness ratio (k_hat) based on the mode.
    3. Check if the actual stiffness ratio (k_hat) meets the required limits.
    4. Calculate minimum required hold-down resistance (Min_rh) based on stiffness, connection resistance, and loading.
    5. Verify if provided hold-down resistance meets minimum requirements and desired kinematic mode.
    """

    # Calculate normalized gravity load effect (q_hat)
    q_hat = q * bs**2 * m_panel**2 / (2*Vf*H)

    # Calculate kinematic mode limits (Min_k_hat, Max_k_hat) for coupled-panel (CP) and single-wall (SW) modes
    Min_k_hat = (1 - q_hat * (3 * m_panel - 2)) / (1 - q_hat * (m_panel - 2))
    Max_k_hat = (1 - q_hat) / (1 - q_hat * (m_panel - 2))

    # Determine actual normalized stiffness (k_hat)
    k_hat = kh / (nf * kf)

    # Determine kinematic mode based on stiffness checks
    if k_hat > Max_k_hat:
        KM = 'CP'  # Coupled-Panel mode
    else:
        KM = 'SW'

    # Determine minimum required hold-down resistance (Min_rh)
    if k_hat < 1:
        Min_rh = max(rf_percentile(rf, COV, 15) * kh / kf, nf * rf_percentile(rf, COV, 15) - q * bs)
    else:
        Min_rh = rf_percentile(rf, COV, 15) * kh / kf

    #Rocking prevented by factored dead loads- Clause 11.9.2.1.2.4
    rck_prv = (q*m_panel*bs**2)/(2*Vf*H)
    rck_check = 1
    if rck_prv < 1 :
        #Calculate Rocking resistance - Clause 11.9.2.1.2.2
        KRPrime = kh + (m_panel-1)*nf*kf #Equivalent hold-down stiffness for CP mode
        KR = KRPrime * bs**2 /(H**2)

        #7) Rwr, Rocking resistance of the CLT shearwall, Clause 11.9.2.1.2.2
        #For CP wall
        Rwr1 = rh*KRPrime*bs/(kh*H) + q*m_panel*bs/(2*H)
        Rwr2 = rf*KRPrime*bs/(kf*H) + q*m_panel*bs/(2*H)
        Rwr = min(Rwr1,Rwr2)

        if Rwr < Mf :
            rck_check = 0

    # Check if the design meets stiffness, resistance, and desired wall behavior criteria
    if rh < Min_rh or KM != WB or rck_check== 0 :
        return 0  # Connection does not meet requirements
    else:
        return 1

def WFS_check(ks, rs, Msr30, Vf, Mf, m_panel, bs, Omega):
    """
    Checks the adequacy of Wall-Floor Shear (WFS) connections for a given CLT shearwall configuration.

    Parameters:
    - ks (float): Stiffness of shear connection in the horizontal direction (kN/m).
    - rs (float): Factored shear resistance of one shear connection (kN).
    - Msr30 (float): 30th percentile of the peak resistance for shear connections (kN·m).
    - Vf (float): Factored lateral force applied at the top of the shearwall (kN).
    - Mf (float): Factored rocking moment acting on the shearwall due to seismic action (kN·m).
    - m_panel (int): Number of CLT panels in a shearwall.
    - bs (float): Length of one CLT panel within the shearwall (m).
    - SR_Margin (float): Margin percentage to ensure connections are not excessively strong.

    Returns:
    - tuple: (int, int)
        - First element is 1 if the shear connection setup is adequate, 0 otherwise.
        - Second element is the required number of shear connections per panel (ns).

    Explanation:
    1. Computes the number of required shear connections per panel (ns).
    2. Validates the calculated number of shear connections:
        - Ensures the shear connection strength does not exceed allowable strength by the specified margin.
        - Ensures minimum spacing of shear connectors is at least 0.3 m (300 mm).

    """
    # Check if shear forces and moments are non-zero to avoid division errors
    if (Vf / m_panel / rs) * Mf != 0:
        # Calculate required number of shear connections per panel
        ns = math.ceil((Msr30 / Mf) * (Vf / m_panel / rs))

        # Check for shear connection over-strength and spacing
        fs_capacity = ns * rs * m_panel
        mltpl_demand = Vf * Omega

        min_spacing = 0.3  # Minimum spacing in meters (300 mm)

        # Calculate spacing check
        spacing_ok = (bs / (ns+1)) >= min_spacing

        # Check if calculated strength is within the allowed margin
        if mltpl_demand > fs_capacity or not spacing_ok:
            return 0, 0
        else:
            return 1, ns

def select_configs(design_models, timeout=300):
    """
    Given a dictionary 'design_models' where keys are storey numbers and values are pandas DataFrames
    with candidate configurations (with an 'Omega' column), this function returns a dictionary with the
    selected configuration for each storey. The chain is chosen so that the ratio (Omega_current / Omega_previous)
    is between 0.9 and 1.2, and for each storey the candidate with the smallest Omega is prioritized.
    
    If the search exceeds `timeout` seconds, the function aborts and returns None.
    
    Parameters:
      design_models (dict): Keys are storey numbers and values are pandas DataFrames of candidate configs.
      timeout (int): Maximum allowed time (in seconds) for the DFS search.
      
    Returns:
      dict or None: A dictionary mapping each storey to its selected configuration (pandas Series), or
                    None if no valid chain is found or timeout is reached.
    """
    # Basic Units
    m = 1.0
    N = 1.0
    mm = m / 1000.0
    kN=1000*N
    Pa = N / (m ** 2)
    MPa = Pa * 1.0e6
    GPa = Pa * 1.0e9

    start_time = time.time()
    storeys = sorted(design_models.keys())
    
    # Sort candidates in each storey by Omega (ascending order)
    sorted_candidates = {}
    for s in storeys:
        sorted_candidates[s] = design_models[s].sort_values(by="Omega").reset_index(drop=True)
    


    # Precompute valid transitions (edges) between consecutive storeys.
    # For each candidate in storey i, record candidate indices in storey i+1 that satisfy:
    #   0.9 <= (Omega_next / Omega_current) <= 1.2.
    edges = {}
    for i in range(len(storeys) - 1):
        current_storey = storeys[i]
        next_storey = storeys[i + 1]
        for j in range(len(sorted_candidates[current_storey])):
            candidate_current = sorted_candidates[current_storey].iloc[j]
            key = (current_storey, j)
            edges[key] = []
            for k in range(len(sorted_candidates[next_storey])):
                candidate_next = sorted_candidates[next_storey].iloc[k]
                ratio = candidate_next["Omega"] / candidate_current["Omega"]
                if 0.9 <= ratio <= 1.2:
                    edges[key].append(k)
    
    # Use memoization to cache results of DFS.
    @lru_cache(maxsize=None)
    def dfs(level, last_candidate):
        # Check for timeout.
        if time.time() - start_time > timeout:
            raise TimeoutError("Timeout reached during DFS search")
        
        # Base case: reached a chain covering all storeys.
        if level == len(storeys):
            return ()
        
        # For the first storey, last_candidate is None.
        if level == 0:
            for j in range(len(sorted_candidates[storeys[0]])):
                res = dfs(level + 1, j)
                if res is not None:
                    return (j,) + res
            return None
        else:
            key = (storeys[level - 1], last_candidate)
            if key in edges:
                for j in edges[key]:
                    res = dfs(level + 1, j)
                    if res is not None:
                        return (j,) + res
            return None

    try:
        solution = dfs(0, None)
    except TimeoutError as e:
        print(e)
        return None

    if solution is not None:
        # Build the resulting dictionary using the found candidate indices.
        selected_configs = {}
        for idx, candidate_index in enumerate(solution):
            storey = storeys[idx]
            candidate = sorted_candidates[storey].iloc[candidate_index]
            selected_configs[storey] = candidate
        return selected_configs
    else:
        return None
    
def Connection_design(models_dict, file_path):
    """
    Generates multiple connection configurations for each model and stores them in a dictionary.

    Parameters:
        models_dict (dict): Dictionary containing model properties and seismic forces.
        Kinematic_mode (str): Desired kinematic mode for the wall.
        Mf (float): Design moment capacity.
        file_path (str): Path to the Excel file with connection data.

    Returns:
        dict: Dictionary containing all configurations for all models.
    """ 
    # Basic Units
    m = 1.0
    N = 1.0
    mm = m / 1000.0
    kN=1000*N
    Pa = N / (m ** 2)
    MPa = Pa * 1.0e6
    GPa = Pa * 1.0e9
       
    COV = 0.2
    SR_Margin = 0.5
    file_path_arch = 'Archtypes_Tier2.xlsx'
    #Load archetypes
    archetypes = pd.read_excel(file_path_arch)

    # file_path='All_Connections.xlsx'
    # file_path='selected_Final.xlsx'
    xls = pd.ExcelFile(file_path)
    sheet_names = ['WFS', 'WFT', 'WWS']

    connections_full_data = {}
    connections = {}

    cols = ['k', 'i', 'j', 'nf', 'ns' ,'Omega' ,'Wall_mech' ,'Zone_labels']

    # Load connection data
    for sheet in sheet_names:
        sheet_data = pd.read_excel(xls, sheet_name='Selected_' + sheet)
        connections_full_data[sheet] = sheet_data
        connections[sheet] = sheet_data[['Ks', 'Δy', 'Fy', 'Δmax', 'Fmax', 'Δu', 'Fu','Zone_label']]

    All_DesignModels=dict()

    # Iterate over all models in the input dictionary
    for model_name, model_data in models_dict.items():
        print('==================================')
        print(f'Model{model_name} starts...')
        occupancy = model_data['Occupancy']
        height_class = model_data['Height Class']
        seismic = model_data['Seismic']
        ductility = model_data['Ductility']
        NumSt = model_data['NumSt']
        H = model_data['H']
        q = model_data['q_value']
        bs = model_data['Panel_width']
        m_panel = model_data['m_panels']
        t = model_data['t']
        WB = model_data['Wall behaviour']
        moments = model_data['Moments']
        shears = model_data['Shears']
        layup = model_data['Layup']

        B = bs * m_panel  # Wall width

        Mf = moments[0]
        Vf = shears[1] 
        Pf = model_data['Pf']
        Pg = model_data['Pg']
        design_models =  dict()
        Count_design_models = dict()

        CLT_properties, Pr, Mr, Vr, Qr, PEV = CLT_resistance_calculator(model_data, file_path='gh-repo-clone-SuStrucSy-CLT-Shearwall-Archetype-Development/CLT Properties.xlsx')
        capacity_check_I_status, capacity_check_I_ratio = Panel_capacity_check_I(Pr, Mr, PEV, Pf[1], t)
        archetypes.at[model_name-1,'Gravity'] = capacity_check_I_status
        if capacity_check_I_status  == 1 :
            for st in range(1 , NumSt+1):
                wall_config = pd.DataFrame(columns=cols)
                design_number = 0

                Mf = moments[st-1]
                Vf = shears[st]      
                
                for k in range(len(connections['WWS'])):
                    # print(f'Storey1 : Trying a WWS Connection : {k}')
                    #Select a WWS connection from the database
                    WWS_data = connections_full_data['WWS'].iloc[k]
                    
                    #Kf & nf
                    kf = WWS_data['Ks'] *kN/m
                    rf = WWS_data['Fy'] * kN
                    nfs = [5,10,15,20,25,30] 
                    for nf in nfs :
                        # print(f'Storey1 : nf : {nf}')
                        for i in range(len(connections['WFT'])):
                            #Select a WFT connection from the database
                            # print(f'Storey1 : Trying a WFT Connection : {i}')
                            WFT_data = connections_full_data['WFT'].iloc[i]
                            
                            #kh & rh
                            kh = WFT_data['Ks'] *kN/m
                            rh = WFT_data['Fy']* kN   

                            WFT_flag = WFT_check (kf,rf,nf,kh,rh,WB,m_panel,bs,H,q,Vf,Mf,COV)
                            if WFT_flag == 0:
                                # print('Failed, Trying with another WFT connection')
                                continue  # Select another WFT connection
                            else:
                                #WFS Connection
                                # print('Moving on to selecting a WFS connection')
                                for j in range(len(connections['WFS'])):
                                    # print(f'Storey1 : Trying a WFS Connection : {j}')
                                    WFS_data = connections_full_data['WFS'].iloc[j]
                                    
                                    #ks & rs
                                    ks = WFS_data['Ks'] *kN/m
                                    rs = WFS_data['Fy'] * kN

                                    KU = 1.0 # KU = 1.0 when uplift contribution of the shear connections is neglected
                                    Msr30 = bs*(rh * KU + rf_percentile(rf,COV, 30)*(m_panel-1)*nf + q*bs*m_panel/2)
                                    # Clause 11.9.3.6.4.2: Omega = Msr95/Mf  for each storey
                                    Msr95 = bs*(rh * KU + rf_percentile(rf, COV, 95)*(m_panel-1)*nf + q*bs*m_panel/2)
                                    Omega_st = Msr95/Mf
                                    WFS_flag,ns = WFS_check (ks,rs,Msr30,Vf,Mf,m_panel,bs,Omega_st)
                                    if WFS_flag == 0:
                                        # print('Failed, Trying with another WFS connection')
                                        continue  # Select another WFS connection
                                        
                                    else:
                                        # print('Connections are Ok for storey1')

                                        #Calculate CLT shearwall mechanical properties

                                        #1) storey over_capacity (Omega)
                                        # Clause 11.9.3.6.4.2: Omega = Msr95/Mf  for each storey

                                        #2) KA, Lateral Stiffness due to shear deformation, Clause 11.9.5.3.1 
                                        GAeff = CLT_properties[int(layup[0])]['GAeff0'] * B * 1e6 * N  # Shear stiffness [N]
                                        KA = GAeff * t * m_panel * bs/H

                                        #3) KB, Lateral stiffness due to bending deformation, Clause 11.9.5.3.2
                                        EIeff0 = CLT_properties[int(layup[0])]['EIeff0'] *B *  1e9 / mm**2 *N
                                        KB = 3*EIeff0/(H**3)

                                        #4) KR, Lateral stiffness due to rocking deformation, Clause 11.9.5.3.3
                                        KRPrime = kh + (m_panel-1)*nf*kf #Equivalent hold-down stiffness for CP mode
                                        KR = KRPrime * bs**2 /(H**2)

                                        #5) Ks, Lateral stiffness to sliding deformation, Clause 11.9.5.3.4
                                        KS = ks * m_panel * ns

                                        #6) Kw, Equivalent lateral stiffness of CLT shearwal, Clause 11.9.5.2
                                        Kw = 1/((1/KA) + (1/KB) + (1/KR) + (1/KS))

                                        #7) Rwr, Rocking resistance of the CLT shearwall, Clause 11.9.2.1.2.2
                                        #For CP wall
                                        Rwr1 = rh*KRPrime*bs/(kh*H) + q*m_panel*bs/(2*H)
                                        Rwr2 = rf*KRPrime*bs/(kf*H) + q*m_panel*bs/(2*H)
                                        Rwr = min(Rwr1,Rwr2)

                                        #Wall Mechanical Properties Matrix
                                        Wall_mech = {'Rwr': Rwr, 'Kw':Kw , 'KS':KS , 'KR':KR, 'KB':KB, 'KA':KA , 'Omega': Omega_st}

                                        Zone_labels = {'WWS' : connections_full_data['WWS'].iloc[k]['Zone_label'] ,
                                                       'WFT' : connections_full_data['WFT'].iloc[i]['Zone_label'] ,
                                                       'WFS' : connections_full_data['WFS'].iloc[j]['Zone_label'] }
                                        
                                        #Panel capacity check II
                                        capacity_check_II_status=Panel_capacity_check_II(Omega_st, Pf[st], Pg[st]/m_panel, Vf/m_panel, Mf/m_panel, Pr, Vr, Mr, Qr)
                                        if capacity_check_II_status == 1:
                                            new_config = pd.DataFrame([{'k': [k], 'i': [i], 'j': [j], 'nf': [nf], 'ns': [ns] ,
                                                                        'Omega': Omega_st , 'Wall_mech' : Wall_mech, 'Zone_labels':Zone_labels}])
                                            design_number += 1
                                            wall_config = pd.concat([wall_config,new_config],axis=0,ignore_index=True)

                                            

                design_models.update({st : wall_config})
                Count_design_models.update({st : len(wall_config)}) 

            selected_configs = select_configs(design_models, timeout=3600)
            if selected_configs is not None :
                DesignModel_data = model_data.copy()  # Copy model data
                DesignModel_data.update({
                            'model': model_name,
                            'Config': selected_configs,
                            'CountConfigs' : Count_design_models,
                            'AllConfigs': design_models

                        })
                
                All_DesignModels.update({model_name:DesignModel_data})
    archetypes.to_excel('archetypes.xlsx')
    return All_DesignModels   




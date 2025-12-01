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


def interpolate_mv_j(file_path, Ta, SRa):
    """
    Interpolates Mv and J values from provided data tables based on input values Ta and SRa,
    following the guidelines from Table 4.1.8.11 of the NRCC 2020.

    Parameters:
    - file_path (str): Path to the Excel file containing sheets named 'Mv' and 'J'.
    - Ta (float): Fundamental period of the building (T), constrained between 0.5 and 5.0.
    - SRa (float): Seismic response spectral acceleration value.

    Returns:
    - tuple: (Interpolated Mv, Interpolated J)
    """
    # Load data from Excel sheets for Mv and J
    mv_data = pd.read_excel(file_path, sheet_name='Mv')
    j_data = pd.read_excel(file_path, sheet_name='J')

    def interpolate_table(data, Ta, SRa):
        """
        Interpolates a single data table for given Ta and SRa values.

        Parameters:
        - data (DataFrame): Table from Excel.
        - Ta (float): Target period.
        - SRa (float): Target spectral acceleration.

        Returns:
        - float: Interpolated result from the table.
        """
        # Extracting T (column headers excluding the first column) and SR values (first column)
        T_values = data.columns[1:].astype(float)
        SR_values = data.iloc[:, 0].values
        table_values = data.iloc[:, 1:].values

        # Ensuring Ta is within valid range (0.5 to 5.0 seconds as per design constraints)
        Ta = np.clip(Ta, 0.5, 5.0)

        # Interpolate each row based on Ta to find intermediate values across SR values
        interpolated_rows = [np.interp(Ta, T_values, table_values[i]) for i in range(len(SR_values))]

        # Interpolate between these intermediate values to find the final result based on SRa
        result = np.interp(SRa, SR_values, interpolated_rows)

        return result

    # Interpolating Mv and J using the interpolation function
    Mv = interpolate_table(mv_data, Ta, SRa)
    J = interpolate_table(j_data, Ta, SRa)

    return Mv, J

# Example usage:
# file_path = 'Wall_system_Mv.xlsx'
# Ta = 1.0  # Target fundamental period (seconds)
# SRa = 0.4  # Target spectral acceleration (g)
# interpolated_Mv, interpolated_J = interpolate_mv_j(file_path, Ta, SRa)
# print(f'Interpolated Mv: {interpolated_Mv}, Interpolated J: {interpolated_J}')


def Interpolate_Sa (T, Sa, Ta):
    """
    Interpolates the spectral acceleration S(Ta) at the fundamental period Ta
    using log-log interpolation based on the provided periods (T) and
    spectral accelerations (Sa).
    Log-Log Interpolation is based on A-4.1.8.4.(6) NRCC 2020

    Parameters:
    T (list of float): List of periods in seconds.
    Sa (list of float): List of spectral acceleration values corresponding to T.
    Ta (float): Fundamental period at which to interpolate S(Ta).

    Returns:
    float: Interpolated spectral acceleration S(Ta).
    """
    if Ta in T:
        Sa_Ta = Sa[T.index(Ta)]
    
    sorted_indices = np.argsort(T)
    T = np.array(T)[sorted_indices]
    Sa = np.array(Sa)[sorted_indices]
    
    if Ta < T[0] or Ta > T[-1]:
        raise ValueError("Ta is outside the range of provided periods.")

        #Table 4.1.8.4.-C NRCC 2020
    if Ta <= 0.2 :
        Sa_Ta = max(Sa[3],Sa[5])
    else:
        for i in range(len(T) - 1):
            if T[i] < Ta < T[i + 1]:
                log_Ti = np.log(T[i])
                log_Ti1 = np.log(T[i + 1])
                log_Sai = np.log(Sa[i])
                log_Sai1 = np.log(Sa[i + 1])
                log_Ta = np.log(Ta)

                log_Sa_Ta = log_Sai + (log_Ta - log_Ti) * (log_Sai1 - log_Sai) / (log_Ti1 - log_Ti)

                Sa_Ta = np.exp(log_Sa_Ta)
                #return Sa_Ta
    # raise ValueError("Failed to interpolate S(Ta). Check input values.")
    return Sa_Ta


def Force_distribution(weight, height, num_storeys, V):
    """
    Calculate lateral forces distributed to each storey for a building with uniform weight and accumulative height.

    Parameters:
        weight (float): Weight (W) of each storey.
        height (float): Height (h) increment for each storey.
        num_storeys (int): Number of storeys in the building.
        V (float): Total seismic base shear force (V).

    Returns:
        list of float: Lateral forces (Fx) distributed to each storey.
    """
    # Total weight-height product for the building
    total_weight_height = sum(weight * (height * (i + 1)) for i in range(num_storeys))

    # Calculate lateral force for each storey
    Fx = [( weight * (height * (i + 1)) / total_weight_height) * V for i in range(num_storeys)]

    return Fx

# Example usage:
# weight = 200  # Weight of each storey in kN
# height = 3  # Height increment per storey in meters
# num_storeys = 3  # Number of storeys in the building
# base_shear = 500  # Total seismic base shear force in kN

# forces = Force_distribution(weight, height, num_storeys, base_shear)
# print("Lateral Forces at each storey:", forces)


def Force_Moment_distribution(weight, height, num_storeys, V, base_j):
    """
    Calculate lateral forces distributed to each storey and overturning moment (Mx) for each storey.(4.1.8.11.(7)&(8) NBCC 2020)

    Parameters:
        weight (float): Weight (W) of each storey.
        height (float): Height (h) increment for each storey.
        num_storeys (int): Number of storeys in the building.
        V (float): Total seismic base shear force (V).
        base_j (float): Base overturning moment reduction factor (J).

    Returns:
        tuple: A tuple containing:
            - list of float: Lateral forces (F) distributed to each storey.
            - list of float: Overturning moments (Mx) for each storey.
    """
    # Calculate total weight-height product for the building
    total_weight_height = sum(weight * (height * (i + 1)) for i in range(num_storeys))

    lateral_forces = {}
    cumulative_force = 0
    for i in range(num_storeys - 1, -1, -1):  # Iterate from top storey to bottom
        storey_force = (weight * (height * (i + 1)) / total_weight_height) * V
        cumulative_force += storey_force
        lateral_forces[i + 1] = cumulative_force

    total_height = num_storeys * height
    
    # Calculate Mx for each storey
    moments = {}
    for x in range(num_storeys):
        hx = (x + 1) * height
        jx = 1.0 if hx >= 0.6 * total_height else base_j + (1 - base_j) * (hx / (0.6 * total_height))
        moment_sum = sum(
            lateral_forces[i + 1] * ((i + 1) * height - hx) for i in range(x, num_storeys)
        )
        mx = jx * moment_sum
        moments[x + 1] = mx

    return lateral_forces, moments


def Force_Moment_distribution(wi, hi, num_storeys, V, base_j,m_panel,bs,q):
    """
    Calculate lateral forces distributed to each storey and overturning moment (Mx) for each storey.

    Parameters:
        wi (float): Weight (W) of each storey.
        hi (float): Height increment (H) for each storey.
        num_storeys (int): Number of storeys in the building.
        V (float): Total seismic base shear force (V).
        base_j (float): Base overturning moment reduction factor (J).

    Returns:
        tuple: A tuple containing:
            - dict: Lateral forces (F_s) distributed to each storey.
            - dict: Overturning moments (Mx) for each storey.
    """
    # Calculate heights of each storey from the ground
    heights = [i * hi for i in range(num_storeys+1)]

    # Calculate Wi*Hi for each storey
    weight_height = [wi * heights[i] for i in range(num_storeys+1)]

    # Calculate total Wi*Hi
    total_weight_height = sum(weight_height)

    # Calculate lateral force for each storey
    lateral_forces = {}
    Pf = {}
    for i in range(num_storeys+1):
        lateral_forces[i] = (weight_height[i] / total_weight_height) * V
        Pf[i] = lateral_forces[i]/m_panel*hi/bs 


    # Total height of the building
    total_height = num_storeys * hi

    # Calculate shear forces (cumulative from top to bottom)
    shear_forces = {}
    Axial_forces_f = {}
    Axial_forces_g = {}
    cumulative_shear = 0
    cumulative_pf = 0
    cumulative_pg = 0
    for i in range(num_storeys , -1, -1):  # Start from the top storey
        cumulative_shear += lateral_forces[i]
        cumulative_pf +=  Pf[i]
        cumulative_pg += wi
        shear_forces[i] = cumulative_shear
        Axial_forces_f[i] = cumulative_pf
        Axial_forces_g[i] = cumulative_pg


    # Calculate Mx for each storey
    moments = {}
    for x in range(num_storeys+1):
        hx = heights[x]
        jx = 1.0 if hx >= 0.6 * total_height else base_j + (1 - base_j) * (hx / (0.6 * total_height))
        moment_sum = sum(
            lateral_forces[i] * (heights[i] - hx) for i in range(x, num_storeys+1)
        )
        mx = jx * moment_sum
        moments[x] = mx

    return lateral_forces, shear_forces, moments, Axial_forces_f , Axial_forces_g


def Seismic_forces_NBCC (wi,hi,num_storeys,duct,IE,T,Sa,m_panel,bs,q):
    W = wi * num_storeys
    h = hi * num_storeys
    S = dict(zip(T, Sa))
    
    #Determine Period (Ta)
    #NBCC 2020 - Clause4.1.8.11 (3)(C) : Ta = 0.05(h)**3/4
    Ta = 0.05*h**0.75

    #Determine The design spectral acceleration S(Ta)
    # Clause 4.1.8.4.(6)
    Sa_Ta = Interpolate_Sa (T, Sa, Ta)

    #Calculate Spectral Ratio S(0.2)/S(5.0)
    SR = S[0.2]/S[5.0]

    #Load Table 4.1.8.11 values for Wall system
    file_path = 'gh-repo-clone-SuStrucSy-CLT-Shearwall-Archetype-Development/Wall system Mv.xlsx'

    Mv, J = interpolate_mv_j(file_path, Ta, SR)

    if duct == 'Moderate' :
        #Moderatelt ductile CLT Shearwall
        Rd , R0 = 2.0 , 1.5
    elif duct == 'Low':
        #Limited ductility CLT Shearwall
        Rd , R0 = 1.0 , 1.3

    #Calculate base shear(4.1.8.11.(2) NRCC 2020)
    # V = S(Ta)MvIEW/(RdR0) 
    C = Sa_Ta*Mv*IE/(Rd*R0)
    V = C*W

    #Calculate Vmin (4.1.8.11.(2-a) NRCC 2020)
    S4 = Interpolate_Sa (T, Sa, 4.0)
    Vmin = S4*Mv*IE*W/(Rd*R0)

    #Calculate Vmax (4.1.8.11.(2-c) NRCC 2020) - For Moderately ductile CLT shearwall 
    Vmax1 = 2/3*S[0.2]*IE*W/(Rd*R0)
    Vmax2 = S[0.5]*IE*W/(Rd*R0)

    if V < Vmin :
        V = Vmin
    if V > min(Vmax1,Vmax2) :
        V = min(Vmax1,Vmax2)

    #Calculate Ft
    if Ta > 0.7 :
        Ft = min(0.07*Ta, 0.25)*V
    else:
        Ft = 0

    #Calculate Storey forces Fx
    forces, shears, moments, Pf, Pg = Force_Moment_distribution(wi, hi, num_storeys,V, J,m_panel,bs,q)  

    return forces, shears , moments , V , Pf, Pg

def Seismic_design (file_path_arch = 'Archtypes_Tier2.xlsx', file_path_hazard = 'Hazard.xlsx') :
    # Basic Units
    m = 1.0
    N = 1.0
    mm = m / 1000.0
    kN=1000*N
    #Load archetypes
    df = pd.read_excel(file_path_arch)

    # Load seismic hazard data
    excel_file = pd.ExcelFile(file_path_hazard )
    hazard = {sheet: excel_file.parse(sheet) for sheet in excel_file.sheet_names}

    # Sa(0.05) [g]	Sa(0.1) [g]	Sa(0.2) [g]	Sa(0.3) [g]	Sa(0.5) [g]	Sa(1.0) [g]	Sa(2.0) [g]	Sa(5.0) [g]	Sa(10.0) [g]	PGA [g]	PGV [m/s]
    T = [0,0.05,0.1,0.2,0.3,0.5,1.0,2.0,5.0,10.0]

    models= dict()
    for i in range(len(df)):
        model = df.iloc[i]
        #Calculate seismic forces
        # Extract parameters from the row
        occupancy = model['Occupancy']
        height_class = model['Height Class']
        seismic = model['Seismic']
        ductility = model['Ductility']
        NumSt = model['NumSt']
        H = model['H']
        q_state = model['q_state']
        q = model['q_value']
        Min_B = model['Min B']
        Max_B = model['Max B']
        total_wall_width = model['Total wall width']
        panel_state = model['Panel_state']
        aspect_ratio = model['Aspect ratio']
        bs = model['Panel_width']
        m_panel = model['m_panels']
        layup = model['Layup']
        t = model['t']
        WB = model['Wall behaviour']
        
        if occupancy[0]=='R':
            IE = 1
            if seismic == 'SC4':
                Sa = hazard['SC4']['F']
            elif seismic == 'SC3':
                Sa = hazard['SC3']['F']
        elif occupancy[0]=='C':
            IE = 1.3
            if seismic == 'SC4':
                Sa = hazard['SC4']['F']
            elif seismic == 'SC3':
                Sa = hazard['SC2']['F']
        else:
            raise ValueError("An error occurred") 

        unit_weight = 5.1*kN/m**3 #According to Nordic catalog
        wall_weight = m_panel*bs*H*t*unit_weight
        wi = q*bs*m_panel  + wall_weight #weight of each storey

        forces , shears , moments , base_shear , Pf, Pg = Seismic_forces_NBCC (wi,H,NumSt,ductility,IE,T,Sa,m_panel,bs,q)

        models[i+1] = {
            'Occupancy': occupancy,
            'Height Class': height_class,
            'Seismic': seismic,
            'Ductility': ductility,
            'NumSt': NumSt,
            'Total Height': H * NumSt,
            'H': H,
            'Wall behaviour' : WB,
            'q_state': q_state,
            'q_value': q,
            'Min B': Min_B,
            'Max B': Max_B,
            'Total wall width': total_wall_width,
            'Panel_state': panel_state,
            'Aspect ratio': aspect_ratio,
            'Panel_width': bs,
            'm_panels': m_panel,
            'Layup': layup,
            't': t,
            'Forces': forces,
            'Shears': shears,
            'Moments': moments,
            'Base Shear': base_shear,
            'Pf' :Pf,
            'Pg' : Pg

        }
    return models



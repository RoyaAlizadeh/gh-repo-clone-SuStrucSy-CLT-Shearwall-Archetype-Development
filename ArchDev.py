import pandas as pd
def Archetype_generator():
    """
    Generates Tier 1 and Tier 2 CLT shearwall archetype configurations for parametric analysis
    based on CSA-O86:24 seismic design provisions.

    Description:
    ------------
    This function systematically creates a library of archetype configurations for Cross-Laminated Timber
    (CLT) wall systems in multi-storey wood buildings. It begins by defining Tier 1 combinations that include:
    - Occupancy types (Residential, Commercial)
    - Height classes (High-rise, Mid-rise, Low-rise)
    - Seismic categories (SC3, SC4)
    - Ductility class (Moderate, non-dissipative)

    From these Tier 1 parameters, the function expands into Tier 2 combinations by evaluating:
    - Panel layup (thickness)
    - Number of storeys
    - Applied line loads (low/high)
    - Panel configuration (Multi-panel)
    - Aspect ratio classes (Low, Mid, High)
    - Width constraints (minimum and maximum bay width)

    The function ensures that all generated combinations meet the seismic height restrictions per CSA-O86,
    and filters out configurations that exceed limits. For each valid configuration, it computes and stores:
    - Wall geometry (panel width, number of panels, total width)
    - Load conditions (line load intensity)
    - Wall behavior classification (Shear Wall or Coupled Panel)
    - CLT layup type and thickness

    Finally, the complete Tier 2 design matrix is exported to an Excel file named `Archtypes_Tier2.xlsx`.

    Returns:
    --------
    combinations_Tier1 : dict
        Dictionary of Tier 1 configurations with unique identifiers.

    combinations_Tier2 : dict
        Dictionary of Tier 2 expanded configurations with geometric and loading properties for structural evaluation.
    """
    # Basic Units
    m = 1.0
    N = 1.0
    mm = m / 1000.0
    kN=1000*N

    # --- Tier 1 Parameters ---
    # Primary occupancy classifications
    Occupancy_types = ['Residential', 'Commercial']

    # Building height classifications
    Height_classes = ['High-rise', 'Mid-rise', 'Low-rise']

    # Seismic design categories as per CSA-O86 Table 4.1.8.4 (referenced indirectly)
    Seismic_categories = ['SC3', 'SC4']

    # Ductility classification (limited to 'Moderate' for non-dissipative CLT walls)
    ductility = ['Moderate']


    # --- Tier 2 Parameters (Derived from Tier 1 filters) ---

    # Maximum allowable height by seismic category [m]
    Height_limit = {
        'SC3': 30 * m,  # Seismic Category 3
        'SC4': 20 * m   # Seismic Category 4
    }

    # Line loads on walls by occupancy and aspect ratio category [kN/m]
    loads = {
        'Residential': {
            'Low': 10 * kN/m,
            'High': 30 * kN/m
        },
        'Commercial': {
            'Low': 40 * kN/m,
            'High': 60 * kN/m
        }
    }

    # Storey height by occupancy type [m]
    Heights = {
        'Residential': 3 * m,
        'Commercial': 4 * m
    }

    # Aspect ratio categories used for layout geometry
    Aspect_ratios = ['Low', 'Mid', 'High']

    # CLT panel width per occupancy and aspect ratio [m]
    panel_width = {
        'Residential': {
            'Low': 1.5 * m,
            'Mid': 1.0 * m,
            'High': 0.75 * m
        },
        'Commercial': {
            'Low': 2.0 * m,
            'Mid': 1.5 * m,
            'High': 1.0 * m
        }
    }

    # Bay width range (min, max) by occupancy type [m]
    Bay_width = {
        'Residential': [0.75 * m, 6 * m],
        'Commercial': [6 * m, 9 * m]
    }

    # Storey range (min, max) for each height class
    numStoreys = {
        'High-rise': [8, 10],
        'Mid-rise': [4, 6],
        'Low-rise': [2, 3]
    }

    # Wall behavior classification by ductility level
    Wall_behavior = {
        'Low': 'SW',         # Shear Wall
        'Moderate': 'CP'     # Coupled Panel
    }

    # CLT panel layup types and their corresponding thicknesses [mm]
    CLT_layups = {
        '3ply': 105 * mm,
        '5ply': 175 * mm,
        '7ply': 265 * mm
    }

    # Panel arrangement options by ductility classification
    panels = {
        'Moderate': ['Multi-panel']
    }

    combinations_Tier1 = {}
    combination_id = 1

    for occupancy in Occupancy_types:
        for height in Height_classes:
            for seismic in Seismic_categories:
                for duct in ductility:  # Note: 'ductility' only contains 'Moderate' in Tier 1
                    # Skip combinations if undesired (e.g., Commercial, SC4, Low ductility) — not relevant here
                    if occupancy == 'Commercial' and seismic == 'SC4' and duct == 'Low':
                        print(f"Excluded: {occupancy}, {seismic}, {duct}, {height}")
                        continue

                    # Store valid combination
                    combinations_Tier1[combination_id] = {
                        'Occupancy': occupancy,
                        'Height': height,
                        'SC': seismic,
                        'Ductility': duct
                    }
                    combination_id += 1

    # Display total number of valid combinations generated
    print(f"Total valid combinations: {len(combinations_Tier1)}")


    combinations_Tier2 = {}
    combination_id = 1

    # Iterate over each Tier 1 configuration
    for key, value in combinations_Tier1.items():
        CC = 0  # Counter to track how many Tier 2 combinations are generated for this Tier 1 case

        # Extract Tier 1 attributes
        height_lim = Height_limit[value['SC']]                 # Maximum height limit for the seismic category
        H = Heights[value['Occupancy']]                        # Storey height based on occupancy
        WB = Wall_behavior[value['Ductility']]                 # Wall behavior based on ductility class

        for layup in CLT_layups:
            t = CLT_layups[layup]  # Panel thickness for the layup

            for i in numStoreys[value['Height']]:  # Iterate over allowed number of storeys
                if H * i <= height_lim:  # Height check
                    NumSt = i
                    qs = loads[value['Occupancy']]  # Load states for given occupancy

                    for q in qs:  # 'q' is the load condition label (e.g., 'Low', 'High')
                        Min_B = Bay_width[value['Occupancy']][0]
                        Max_B = Bay_width[value['Occupancy']][1]

                        for panel_state in panels[value['Ductility']]:
                            for AR in Aspect_ratios:
                                bs = panel_width[value['Occupancy']][AR]  # Single panel width

                                if panel_state == 'Single-panel':
                                    # Case for one panel only
                                    m_panels = 1
                                    B = m_panels * bs
                                    # Store Tier 2 configuration
                                    combinations_Tier2[combination_id] = {
                                        'Combination Tier1': key,
                                        'Occupancy': value['Occupancy'],
                                        'Height Class': value['Height'],
                                        'Seismic': value['SC'],
                                        'Ductility': value['Ductility'],
                                        'NumSt': NumSt,
                                        'Total Height': H * NumSt,
                                        'H': H,
                                        'Wall behaviour': WB,
                                        'q_state': q,
                                        'q_value': qs[q],
                                        'Min B': Min_B,
                                        'Max B': Max_B,
                                        'Total wall width': B,
                                        'Panel_state': panel_state,
                                        'Aspect ratio': AR,
                                        'Panel_width': bs,
                                        'm_panels': m_panels,
                                        'Layup': layup,
                                        't': t
                                    }
                                    combination_id += 1
                                    CC += 1

                                else:
                                    # Case for multiple panels (2 to 5 panels)
                                    for m_panels in range(2, 6):
                                        B = m_panels * bs
                                        if Min_B <= B <= Max_B:
                                            combinations_Tier2[combination_id] = {
                                                'Combination Tier1': key,
                                                'Occupancy': value['Occupancy'],
                                                'Height Class': value['Height'],
                                                'Seismic': value['SC'],
                                                'Ductility': value['Ductility'],
                                                'NumSt': NumSt,
                                                'Total Height': H * NumSt,
                                                'H': H,
                                                'Wall behaviour': WB,
                                                'q_state': q,
                                                'q_value': qs[q],
                                                'Min B': Min_B,
                                                'Max B': Max_B,
                                                'Total wall width': B,
                                                'Panel_state': panel_state,
                                                'Aspect ratio': AR,
                                                'Panel_width': bs,
                                                'm_panels': m_panels,
                                                'Layup': layup,
                                                't': t
                                            }
                                            combination_id += 1
                                            CC += 1

    df = pd.DataFrame(combinations_Tier2).T
    df.to_excel('Archtypes_Tier2.xlsx')

    return combinations_Tier1 , combinations_Tier2

Archetype_generator()
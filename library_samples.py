from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

'''This file is a library containing general informations on the samples collected
   Must be completed 
'''

# ~~~~~~~~~~~~~~~ TABLE OF CONTENTS ~~~~~~~~~~~~~~~~ #
#       0.      CLASS DEFINITION                l. 14
#       1.      Bi-CUPRATES                     l. 42
#       2.      Hg-CUPRATES                     l. 108
#       3.      Sr-RUTENATES                    l. 138
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

#_________________ 0 _________________#

########## CLASS DEFINITION ###########
#_____________________________________#

@dataclass
class TightBindingParamsCu:
    mu: float
    t0: float
    t1: float
    t2: float
    t3: float
    tbi: float

@dataclass
class Doping:
    label: str       # "UD", "OP", "OD"...
    p: float         # doping level (per CuO2)
    Tc: float        # critical temperature
    tb_params: TightBindingParamsCu

@dataclass
class Material:
    name: str
    formula: str
    lattice: Tuple[float, float, float]  # (a, b, c)
    layers: int
    Tc_max : float
    dopings: Dict[str, Doping] = field(default_factory=dict)
    notes: str =""

#_________________ 1 _________________#

############# Bi-CUPRATES #############
#_____________________________________#

Bi2201 = Material(
    name="Bi2201",
    formula="Bi2Sr2CuO6+δ",
    lattice=(5.36, 5.36, 24.6),
    layers=1,
    Tc_max = 35,
    dopings={
        "UD20": Doping("UD20", p=0.10, Tc=20,
                       tb_params=None),
        "OP": Doping("OP", p=0.16, Tc=35,
                       tb_params=None),
        "OD15": Doping("OD15", p=0.22, Tc=15, 
                       tb_params=None),
    },
    notes="Single-layer cuprate, Tc max ~35 K."
)

Bi2212 = Material(
    name="Bi2212",
    formula="Bi2Sr2CaCu2O8+δ",
    lattice=(3.82, 3.82, 30.8),
    layers=2,
    Tc_max = 95,
    dopings={
        "UD75": Doping(
            "UD75", p=0.12, Tc=75,
            tb_params=TightBindingParamsCu(
                mu=-0.18, t0=0.36, t1=-0.10, t2=0.08, ## as example has to be changed
                t3=-0.02, tbi=0.11
            )
        ),
        "OP": Doping(
            "OP", p=0.16, Tc=95,
            tb_params=TightBindingParamsCu(
                mu=-0.15, t0=0.36, t1=-0.11, t2=0.08, ## as example has to be changed
                t3=-0.03, tbi=0.12
            )
        ),
        "OD65": Doping(
            "OD65", p=0.21, Tc=65,
            tb_params=TightBindingParamsCu(
                mu=-0.12, t0=0.36, t1=-0.12, t2=0.07, ## as example has to be changed
                t3=-0.04, tbi=0.10
            )
        ),
    },
    notes="Bilayer cuprate, Tc max ~95 K"
)

Bi2223 = Material(
    name="Bi2223",
    formula="Bi2Sr2Ca2Cu3O10+δ",
    lattice=(3.82, 3.82, 37.1),
    layers=3,
    Tc_max = 110, 
    dopings={
        "UD90": Doping("UD90", p=0.12, Tc=90, tb_params=None),
        "OP": Doping("OP", p=0.16, Tc=110, tb_params=None),
        "OD80": Doping("OD80", p=0.20, Tc=80, tb_params=None),
    },
    notes="Trilayer cuprate, Tc max ~110 K."
)

#_________________ 2 _________________#

############# Hg-CUPRATES #############
#_____________________________________#

Hg1223 = Material(
    name="Hg1223",
    formula="HgBa2Ca2Cu3O8+δ",
    lattice=(3.85, 3.85, 30.8),
    layers=3,
    Tc_max = 134,
    dopings={
        "UD110": Doping(
            "UD110", p=0.12, Tc=110,
            tb_params=TightBindingParamsCu(
                mu=-0.20, t0=0.40, t1=-0.12, t2=0.09, ## as example has to be changed
                t3=-0.03, tbi=0.13
            )
        ),
        "OP": Doping(
            "OP", p=0.20, Tc=134,
            tb_params=TightBindingParamsCu(
                mu=-0.15, t0=0.40, t1=-0.14, t2=0.09, ## as example has to be changed
                t3=-0.05, tbi=0.15
            )
        ),
    },
    notes="Trilayer cuprate with Tc up to 134 K at ambient pressure. Exhibits the highest Tc among cuprates at ambient pressure."
)

#_____________________________________#

############# Sr-RUTENATES ############
#_____________________________________#

Sr214 = Material(
    name="Sr214",
    formula="Sr2RuO4",
    lattice=(3.87, 3.87, 12.7),
    Tc_max= 1,
    layers=1,
    notes="Layered ruthenate, unconventional superconductor (p-wave candidate)."
)

Sr327 = Material(
    name="Sr327",
    formula="Sr3Ru2O7",
    Tc_max = None,
    lattice=(3.87, 3.87, 20.7),
    layers=2,
    notes="Bilayer ruthenate, metamagnetic quantum criticality."
)

#_____________________________________#

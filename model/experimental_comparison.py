from pyrealm.pmodel import (
    SubdailyScaler,
    memory_effect,
    SubdailyPModel,
    PModelEnvironment,
    PModel,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyrealm.pmodel.optimal_chi import OptimalChiPrentice14
from pyrealm.pmodel.functions import calc_ftemp_arrh, calc_ftemp_kphio,calc_modified_arrhenius_factor,calc_gammastar,calc_kmm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyrealm.constants import PModelConst
from pyrealm.core.pressure import calc_patm
from pyrealm.pmodel.quantum_yield import QuantumYieldSandoval
from pyrealm.core.hygro import convert_rh_to_vpd
# 👇 define correct sandoval function
from pyrealm.pmodel.quantum_yield  import QuantumYieldABC
import numpy as np
from warnings import warn

class QuantumYieldSandoval(
    QuantumYieldABC,
    method="sandoval",
    requires=["aridity_index", "mean_growth_temperature"],
    default_reference_kphio=1.0 / 9.0,   # ✅ 必须加上这个
    array_reference_kphio_ok=False
):
    """Overridden Sandoval-style quantum yield using Medlyn-style Arrhenius."""

    __experimental__: bool = True

    def peak_quantum_yield(self, aridity_index: np.ndarray) -> np.ndarray:
        phi0_theo = 1.0 / 9.0
        m = 4.090556
        n = 0.121122
        return phi0_theo / (1 + aridity_index**m)**n

    def _calculate_kphio(self) -> None:
        warn("QuantumYieldSandoval (overridden) is experimental.")

        ai = self.env.aridity_index
        mgdd = self.env.mean_growth_temperature
        tcleaf = self.env.tc
        tkleaf = self.env.tc + 273.15
        R = self.env.core_const.k_R

        dS0 = 3468.185
        dS_mgdd = 0.6680158
        Ha = 70885.39

        DeltaS = dS0 * mgdd**(-dS_mgdd)
        Hd = 295.0 * DeltaS

        Top = Hd / (DeltaS - R * np.log(Ha / (Hd - Ha)))

        def medlyn_peaked_arrhenius(tk, Top, Ha, Hd, dS):
            f1 = Hd * np.exp(Ha * (tk - Top) / (tk * R * Top))
            f2 = Hd - Ha * (1 - np.exp(Hd * (tk - Top) / (tk * R * Top)))
            return f1 / f2

        phi0_peak = self.peak_quantum_yield(ai)
        phi0_fT = medlyn_peaked_arrhenius(tkleaf, Top, Ha, Hd, DeltaS)

        self.kphio = phi0_peak * phi0_fT
QuantumYieldSandoval = QuantumYieldSandoval

filling_vpd = pd.read_csv("/Users/yongzhe/Documents/Reading/meta analysis c star/tpc_data_cstar_sensitivity_suffix.csv")
filling_vpd['Thome'] = 'NA'

filling_vpd.loc[filling_vpd["taxon"] == "BOOF", "Thome"] = 21.50532
filling_vpd.loc[filling_vpd["taxon"] == "HOVU", "Thome"] = 21.23116
filling_vpd.loc[filling_vpd["taxon"] == "RASA", "Thome"] = 23.83171
from dataclasses import replace
from pyrealm.constants import PModelConst
ha_vcmax25 = 65330  # J mol^-1
ha_jmax25  = 43900  # J mol^-1
Tref = 298.15
R = 8.314
# ============================================================
# define range of cstar
# ============================================================
base_const = PModelConst()
base_cstar = base_const.wang17_c
base_cstar = 0.41
delta_abs = 0.112
c_star_values = [
    base_cstar - delta_abs,
    base_cstar,
    base_cstar + delta_abs
]

# 基础 DataFrame（复制一次）
df_out = filling_vpd.copy()

for c_star in c_star_values:
    print(f"\n=== Testing c_star = {c_star:.5f} ===")
    custom_const = replace(base_const, wang17_c=c_star)

    gpp_results = []
    anet_results = []
    ci_results = []
    ac_results = []
    aj_results = []
    vcmax_results = []
    jmax_results = []
    rd_results = []
    jmax25_results = []
    vcmax25_results = []
    for idx, row in filling_vpd.iterrows():
        acclim_temp = row["temp_acclim"]
        co2_acclim = row["co2_acclim"]
        patm_acclim = calc_patm(row['z'])
        vpd_acclim = row["vpd_acclim"]
        ppfd_acclim = row["light_acclim"]
        mean_growth_temp  = row["Thome"]
        env_acclim = PModelEnvironment(
            tc=np.array([acclim_temp]),
            patm=np.array([patm_acclim]),
            vpd=np.array([vpd_acclim]),
            co2=np.array([co2_acclim]),
            pmodel_const=custom_const
        )
        env_acclim.aridity_index = 0
        env_acclim.mean_growth_temperature = np.array([mean_growth_temp])

        pmodel_acclim = PModel(env=env_acclim,
                               method_kphio="sandoval",
                               reference_kphio='1.0/9.0',
                               method_optchi="prentice14")
        pmodel_acclim.estimate_productivity(fapar=1, ppfd=np.array([ppfd_acclim]))

        vcmax25 = pmodel_acclim.vcmax[0] * np.exp((ha_vcmax25/R)*(1/(acclim_temp+273.15)-1/Tref))
        jmax25 = pmodel_acclim.jmax[0] * np.exp((ha_jmax25/R)*(1/(acclim_temp+273.15)-1/Tref))

        tc = row["Tleaf"]
        patm = patm_acclim
        vpd = row["VPDleaf"] * 1000
        co2 = row["CO2_r"]
        ppfd = row["Qin"]

        env_subdaily = PModelEnvironment(
            tc=np.array([tc]),
            patm=np.array([patm]),
            vpd=np.array([vpd]),
            co2=np.array([co2]),
            pmodel_const=custom_const
        )
        env_subdaily.aridity_index = 0
        env_subdaily.mean_growth_temperature = np.array([mean_growth_temp])
        model = PModel(env_subdaily,
                       method_kphio="sandoval",
                       reference_kphio='1.0/9.0',
                       method_optchi="prentice14")

        vcmax = vcmax25 * np.exp((ha_vcmax25/R)*(1/Tref - 1/(tc + 273.15)))
        jmax = jmax25 * np.exp((ha_jmax25/R)*(1/Tref - 1/(tc + 273.15)))

        sub_chi = OptimalChiPrentice14(env=env_subdaily)
        xi = pmodel_acclim.optchi.xi
        sub_chi.estimate_chi(xi_values=xi)
        ci = sub_chi.ci

        gammastar = calc_gammastar(tc, patm)
        kmm = calc_kmm(tc, patm)

        ac = vcmax * (ci - gammastar) / (ci + kmm)
        fkphio = model.kphio.kphio
        J = (4 * fkphio * ppfd) / np.sqrt(1 + (4 * fkphio * ppfd / jmax) ** 2)
        aj = (J / 4) * ((ci - gammastar) / (ci + 2*gammastar))

        gpp = min(ac[0], aj[0])
        rd = 0.015 * vcmax
        anet = gpp - rd

        gpp_results.append(gpp)
        anet_results.append(anet)
        ci_results.append(ci[0])
        ac_results.append(ac[0])
        aj_results.append(aj[0])
        vcmax_results.append(vcmax)
        jmax_results.append(jmax)
        rd_results.append(rd)
        jmax25_results.append(jmax25)
        vcmax25_results.append(vcmax25)

    # ============================================================
    # output
    # ============================================================
    suffix = f"_cstar{c_star:.5f}"
    df_out[f"GPP_pmodel{suffix}"] = np.clip(gpp_results, 0, None)
    df_out[f"anet_pmodel{suffix}"] = anet_results
    df_out[f"ci_pmodel{suffix}"] = ci_results
    df_out[f"ac_pmodel{suffix}"] = ac_results
    df_out[f"aj_pmodel{suffix}"] = aj_results
    df_out[f"vcmax_pmodel{suffix}"] = vcmax_results
    df_out[f"jmax_pmodel{suffix}"] = jmax_results
    df_out[f"rd_pmodel{suffix}"] = rd_results
    df_out[f"jmax25_pmodel{suffix}"] = jmax25_results
    df_out[f"vcmax25_pmodel{suffix}"] = vcmax25_results

# save
df_out.to_csv("tpc_data_cstar_sensitivity_suffix_new_phi0.csv", index=False)
print("✅ Saved: tpc_data_cstar_sensitivity_suffix.csv")

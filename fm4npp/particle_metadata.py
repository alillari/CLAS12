"""Particle labels and charges from signed PDG IDs, without a runtime database.

Evaluation classes are pion=0, kaon=1, proton=2, electron=3, other=4.
These are the evaluation YAML's classes, not the classifier training indices.
Unknown IDs retain an unknown charge; the sign of a PDG ID is not a charge.
Reference: https://pdg.lbl.gov/2022/reviews/rpp2022-rev-monte-carlo-numbering.pdf
"""
import math


EVALUATION_PID_LABELS = {0: "pion", 1: "kaon", 2: "proton", 3: "electron", 4: "other"}
_CLASSES = {211: 0, 321: 1, 2212: 2, 11: 3}
# Electric charge in units of e for the positive PDG code. Antiparticles reverse it.
_CHARGES = {
    11: -1, 13: -1, 15: -1, 12: 0, 14: 0, 16: 0, 22: 0,
    111: 0, 211: 1, 221: 0, 331: 0, 130: 0, 310: 0, 311: 0, 321: 1,
    2112: 0, 2212: 1, 3122: 0, 3112: -1, 3212: 0, 3222: 1,
    3312: -1, 3322: 0, 3334: -1,
}


def metadata_from_pdg(value):
    """Return JSON-safe metadata for a PDG ID; missing/unknown never mean +1."""
    try:
        number = float(value)
        valid = not isinstance(value, bool) and math.isfinite(number) and number.is_integer()
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid or number in (0, -100):
        return dict(pdg=None, pid_class=-1, pid_label="unknown", charge=None,
                    pid_metadata_status="missing", charge_source="pdg")
    pdg = int(number)
    code = abs(pdg)
    charge = _CHARGES.get(code)
    # PDG ion format: 10LZZZAAAI (Z protons, A baryons, L Lambdas).
    if charge is None and code // 100000000 == 10:
        strange, protons, baryons = (code // 10000000) % 10, (code // 10000) % 1000, (code // 10) % 1000
        if baryons > 0 and protons + strange <= baryons:
            charge = protons
    if charge is not None and pdg < 0:
        charge = -charge
    category = _CLASSES.get(code, 4)
    return dict(pdg=pdg, pid_class=category, pid_label=EVALUATION_PID_LABELS[category],
                charge=charge, pid_metadata_status="known" if charge is not None else "unknown",
                charge_source="pdg")


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pid', type=int, nargs='+', help='Signed PDG particle IDs')
    print(json.dumps([metadata_from_pdg(pid) for pid in parser.parse_args().pid], indent=2))

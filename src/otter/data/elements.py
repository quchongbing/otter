"""
otter/data/elements.py

Periodic-table lookup utilities used by tests and example scripts.

Design goals
------------
- Keep one canonical mapping: atomic number (Z), symbol, atomic mass.
- Allow case-insensitive symbol lookup.
- Provide simple helpers for converting between Z/symbol/mass.

Notes
-----
Atomic masses are standard atomic weights for stable elements and conventional
representative values for short-lived/transuranic elements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


@dataclass(frozen=True)
class Element:
    """Periodic-table entry."""

    z: int
    symbol: str
    atomic_mass: float


_ELEMENT_ROWS: Tuple[Tuple[int, str, float], ...] = (
    (1, "H", 1.008),
    (2, "He", 4.002602),
    (3, "Li", 6.94),
    (4, "Be", 9.0121831),
    (5, "B", 10.81),
    (6, "C", 12.011),
    (7, "N", 14.007),
    (8, "O", 15.999),
    (9, "F", 18.998403163),
    (10, "Ne", 20.1797),
    (11, "Na", 22.98976928),
    (12, "Mg", 24.305),
    (13, "Al", 26.9815385),
    (14, "Si", 28.085),
    (15, "P", 30.973761998),
    (16, "S", 32.06),
    (17, "Cl", 35.45),
    (18, "Ar", 39.948),
    (19, "K", 39.0983),
    (20, "Ca", 40.078),
    (21, "Sc", 44.955908),
    (22, "Ti", 47.867),
    (23, "V", 50.9415),
    (24, "Cr", 51.9961),
    (25, "Mn", 54.938044),
    (26, "Fe", 55.845),
    (27, "Co", 58.933194),
    (28, "Ni", 58.6934),
    (29, "Cu", 63.546),
    (30, "Zn", 65.38),
    (31, "Ga", 69.723),
    (32, "Ge", 72.63),
    (33, "As", 74.921595),
    (34, "Se", 78.971),
    (35, "Br", 79.904),
    (36, "Kr", 83.798),
    (37, "Rb", 85.4678),
    (38, "Sr", 87.62),
    (39, "Y", 88.90584),
    (40, "Zr", 91.224),
    (41, "Nb", 92.90637),
    (42, "Mo", 95.95),
    (43, "Tc", 98.0),
    (44, "Ru", 101.07),
    (45, "Rh", 102.90550),
    (46, "Pd", 106.42),
    (47, "Ag", 107.8682),
    (48, "Cd", 112.414),
    (49, "In", 114.818),
    (50, "Sn", 118.710),
    (51, "Sb", 121.760),
    (52, "Te", 127.60),
    (53, "I", 126.90447),
    (54, "Xe", 131.293),
    (55, "Cs", 132.90545196),
    (56, "Ba", 137.327),
    (57, "La", 138.90547),
    (58, "Ce", 140.116),
    (59, "Pr", 140.90766),
    (60, "Nd", 144.242),
    (61, "Pm", 145.0),
    (62, "Sm", 150.36),
    (63, "Eu", 151.964),
    (64, "Gd", 157.25),
    (65, "Tb", 158.92535),
    (66, "Dy", 162.500),
    (67, "Ho", 164.93033),
    (68, "Er", 167.259),
    (69, "Tm", 168.93422),
    (70, "Yb", 173.045),
    (71, "Lu", 174.9668),
    (72, "Hf", 178.49),
    (73, "Ta", 180.94788),
    (74, "W", 183.84),
    (75, "Re", 186.207),
    (76, "Os", 190.23),
    (77, "Ir", 192.217),
    (78, "Pt", 195.084),
    (79, "Au", 196.966569),
    (80, "Hg", 200.592),
    (81, "Tl", 204.38),
    (82, "Pb", 207.2),
    (83, "Bi", 208.98040),
    (84, "Po", 209.0),
    (85, "At", 210.0),
    (86, "Rn", 222.0),
    (87, "Fr", 223.0),
    (88, "Ra", 226.0),
    (89, "Ac", 227.0),
    (90, "Th", 232.0377),
    (91, "Pa", 231.03588),
    (92, "U", 238.02891),
    (93, "Np", 237.0),
    (94, "Pu", 244.0),
    (95, "Am", 243.0),
    (96, "Cm", 247.0),
    (97, "Bk", 247.0),
    (98, "Cf", 251.0),
    (99, "Es", 252.0),
    (100, "Fm", 257.0),
    (101, "Md", 258.0),
    (102, "No", 259.0),
    (103, "Lr", 266.0),
    (104, "Rf", 267.0),
    (105, "Db", 268.0),
    (106, "Sg", 269.0),
    (107, "Bh", 270.0),
    (108, "Hs", 277.0),
    (109, "Mt", 278.0),
    (110, "Ds", 281.0),
    (111, "Rg", 282.0),
    (112, "Cn", 285.0),
    (113, "Nh", 286.0),
    (114, "Fl", 289.0),
    (115, "Mc", 290.0),
    (116, "Lv", 293.0),
    (117, "Ts", 294.0),
    (118, "Og", 294.0),
)

_BY_Z: Dict[int, Element] = {z: Element(z=z, symbol=symbol, atomic_mass=mass) for z, symbol, mass in _ELEMENT_ROWS}
_BY_SYMBOL: Dict[str, Element] = {item.symbol.upper(): item for item in _BY_Z.values()}


def _normalize_symbol(symbol: str) -> str:
    """Return canonical case-insensitive symbol key."""
    s = str(symbol).strip()
    if not s:
        raise ValueError("Element symbol cannot be empty.")
    return s.upper()


def all_elements() -> Tuple[Element, ...]:
    """Return all element entries ordered by atomic number."""
    return tuple(_BY_Z[z] for z in sorted(_BY_Z))


def element_from_z(z: int) -> Element:
    """Return Element for atomic number z."""
    z_int = int(z)
    if z_int not in _BY_Z:
        raise KeyError(f"Unknown atomic number: {z_int}")
    return _BY_Z[z_int]


def element_from_symbol(symbol: str) -> Element:
    """Return Element for chemical symbol (case-insensitive)."""
    key = _normalize_symbol(symbol)
    if key not in _BY_SYMBOL:
        raise KeyError(f"Unknown element symbol: {symbol}")
    return _BY_SYMBOL[key]


def element(key: int | str) -> Element:
    """Return Element from atomic number or symbol string."""
    if isinstance(key, int):
        return element_from_z(key)
    if isinstance(key, str):
        s = key.strip()
        if s.isdigit():
            return element_from_z(int(s))
        return element_from_symbol(s)
    raise TypeError("Element key must be int (Z) or str (symbol).")


def atomic_number(symbol: str) -> int:
    """Return atomic number for symbol."""
    return element_from_symbol(symbol).z


def atomic_symbol(z: int) -> str:
    """Return symbol for atomic number."""
    return element_from_z(z).symbol


def atomic_weight(key: int | str) -> float:
    """Return atomic mass (atomic weight) for atomic number or symbol."""
    return element(key).atomic_mass


def has_element(key: int | str) -> bool:
    """Return True if key is a known element symbol or atomic number."""
    try:
        element(key)
        return True
    except (TypeError, ValueError, KeyError):
        return False


__all__ = [
    "Element",
    "all_elements",
    "element",
    "element_from_z",
    "element_from_symbol",
    "atomic_number",
    "atomic_symbol",
    "atomic_weight",
    "has_element",
]


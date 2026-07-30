"""Elementary (1D) Cellular Automaton using Wolfram's numbering scheme.

This module implements the 256 elementary cellular automata where each cell
has two states (0 or 1) and the next state depends on the cell and its two
immediate neighbors (a 3-cell neighborhood).

Classes
-------
ElementaryCA
    A configurable 1D cellular automaton that can run and visualize itself.

Examples
--------
>>> from automaton.elementary import ElementaryCA
>>> ca = ElementaryCA(rule=30, width=101, steps=50)
>>> ca.run()
>>> ca.grid.shape
(51, 101)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass
class ElementaryCA:
    """A 1D elementary cellular automaton using Wolfram's rule numbering.

    The rule number (0-255) encodes the output for all 8 possible 3-cell
    neighborhood patterns. The binary representation of the rule number
    maps each pattern to its output:

        Pattern index: 111 110 101 100 011 010 001 000
        Rule 30 bits:   0   0   0   1   1   1   1   0

    So for Rule 30: pattern '111' -> 0, '110' -> 0, ..., '001' -> 1, '000' -> 0.

    Parameters
    ----------
    rule : int
        Wolfram rule number (0-255).
    width : int
        Number of cells in each row.
    steps : int
        Number of generations to simulate.
    init : str
        Initial condition: "single" (one cell in center), "random",
        or "custom" (provide initial_state).
    initial_state : NDArray | None
        Custom initial state array of shape (width,) with values 0 or 1.
        Only used when init="custom".
    boundary : str
        Boundary condition: "wrap" (periodic) or "zero" (dead cells outside).

    Attributes
    ----------
    grid : NDArray[np.uint8]
        2D array of shape (steps+1, width) holding the full history.
        Row 0 is the initial state, row -1 is the final generation.
    ruleset : NDArray[np.uint8]
        The 8-element lookup table derived from the rule number.
    """

    rule: int = 30
    width: int = 101
    steps: int = 50
    init: Literal["single", "random", "custom"] = "single"
    initial_state: NDArray[np.uint8] | None = field(default=None, repr=False)
    boundary: Literal["wrap", "zero"] = "wrap"

    # Computed after __post_init__
    grid: NDArray[np.uint8] = field(init=False, repr=False)
    ruleset: NDArray[np.uint8] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate parameters and build the ruleset lookup table."""
        if not 0 <= self.rule <= 255:
            raise ValueError(f"Rule must be 0-255, got {self.rule}")
        if self.width < 3:
            raise ValueError(f"Width must be >= 3, got {self.width}")
        if self.steps < 1:
            raise ValueError(f"Steps must be >= 1, got {self.steps}")

        # Build ruleset: index i gives the output for pattern with value i
        # Pattern value: left*4 + center*2 + right (binary 0-7)
        self.ruleset = np.array(
            [(self.rule >> i) & 1 for i in range(8)], dtype=np.uint8
        )

        # Initialize grid
        self.grid = np.zeros((self.steps + 1, self.width), dtype=np.uint8)
        self._init_first_row()

    def _init_first_row(self) -> None:
        """Set up the initial state (row 0 of the grid)."""
        if self.init == "single":
            self.grid[0, self.width // 2] = 1
        elif self.init == "random":
            rng = np.random.default_rng()
            self.grid[0] = rng.integers(0, 2, size=self.width, dtype=np.uint8)
        elif self.init == "custom":
            if self.initial_state is None:
                raise ValueError("init='custom' requires initial_state array")
            state = np.asarray(self.initial_state, dtype=np.uint8)
            if state.shape != (self.width,):
                raise ValueError(
                    f"initial_state shape {state.shape} doesn't match width {self.width}"
                )
            self.grid[0] = state

    def _get_neighborhood(self, row: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Compute the 3-bit neighborhood index for every cell in a row.

        Parameters
        ----------
        row : NDArray[np.uint8]
            Current generation, shape (width,).

        Returns
        -------
        NDArray[np.uint8]
            Array of shape (width,) where each value is 0-7 representing
            the neighborhood pattern (left*4 + center*2 + right).
        """
        if self.boundary == "wrap":
            left = np.roll(row, 1)
            right = np.roll(row, -1)
        else:  # zero boundary
            left = np.zeros_like(row)
            right = np.zeros_like(row)
            left[1:] = row[:-1]
            right[:-1] = row[1:]

        return (left.astype(np.uint8) * 4
                + row.astype(np.uint8) * 2
                + right.astype(np.uint8))

    def step(self, row: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Compute the next generation from a single row.

        Parameters
        ----------
        row : NDArray[np.uint8]
            Current generation, shape (width,).

        Returns
        -------
        NDArray[np.uint8]
            Next generation, shape (width,).
        """
        indices = self._get_neighborhood(row)
        return self.ruleset[indices]

    def run(self) -> NDArray[np.uint8]:
        """Run the automaton for all steps, filling self.grid.

        Returns
        -------
        NDArray[np.uint8]
            The complete grid of shape (steps+1, width).
        """
        for t in range(self.steps):
            self.grid[t + 1] = self.step(self.grid[t])
        return self.grid

    def reset(self, rule: int | None = None, init: str | None = None) -> None:
        """Reset the automaton, optionally changing rule or init mode.

        Parameters
        ----------
        rule : int | None
            New rule number. If None, keeps the current rule.
        init : str | None
            New init mode. If None, keeps the current mode.
        """
        if rule is not None:
            if not 0 <= rule <= 255:
                raise ValueError(f"Rule must be 0-255, got {rule}")
            self.rule = rule
            self.ruleset = np.array(
                [(self.rule >> i) & 1 for i in range(8)], dtype=np.uint8
            )
        if init is not None:
            self.init = init  # type: ignore[assignment]

        self.grid = np.zeros((self.steps + 1, self.width), dtype=np.uint8)
        self._init_first_row()

    def plot(self, ax=None, cmap: str = "binary", title: str | None = None):
        """Plot the automaton grid using matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes | None
            Axes to plot on. If None, creates a new figure.
        cmap : str
            Matplotlib colormap name. Default "binary" (black and white).
        title : str | None
            Plot title. If None, uses "Rule {rule}".

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        ax.imshow(self.grid, cmap=cmap, interpolation="nearest", aspect="auto")
        ax.set_title(title or f"Rule {self.rule}", fontsize=14)
        ax.set_xlabel("Cell")
        ax.set_ylabel("Generation")
        ax.set_yticks([])
        ax.set_xticks([])

        return ax

    @staticmethod
    def rule_table_str(rule: int) -> str:
        """Return a human-readable string showing the rule lookup table.

        Parameters
        ----------
        rule : int
            Wolfram rule number (0-255).

        Returns
        -------
        str
            Formatted string showing each 3-cell pattern and its output.
        """
        patterns = ["111", "110", "101", "100", "011", "010", "001", "000"]
        bits = [(rule >> (7 - i)) & 1 for i in range(8)]
        lines = ["Pattern  ->  Output"]
        lines.append("-" * 20)
        for pat, bit in zip(patterns, bits):
            lines.append(f"  {pat}    ->    {bit}")
        return "\n".join(lines)

    @property
    def density(self) -> NDArray[np.float64]:
        """Fraction of live cells in each generation.

        Returns
        -------
        NDArray[np.float64]
            Array of shape (steps+1,) with density per row.
        """
        return self.grid.mean(axis=1).astype(np.float64)

    @property
    def wolfram_class(self) -> int:
        """Estimate the Wolfram classification (1-4) of this rule.

        This is a rough heuristic based on density evolution:
        - Class 1: converges to uniform (all 0 or all 1)
        - Class 2: converges to periodic/stable patterns
        - Class 3: chaotic, density stays near 0.5
        - Class 4: complex, localized structures (hardest to detect)

        Returns
        -------
        int
            Estimated Wolfram class (1, 2, 3, or 4).
        """
        d = self.density
        final_density = d[-10:].mean() if len(d) >= 10 else d[-1]
        density_var = d[-20:].var() if len(d) >= 20 else d.var()

        if final_density < 0.01 or final_density > 0.99:
            return 1
        if density_var < 0.0001:
            return 2
        if 0.3 < final_density < 0.7 and density_var < 0.005:
            return 3
        return 4  # uncertain, could be complex

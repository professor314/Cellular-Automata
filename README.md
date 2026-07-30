# 1D Cellular Automata

An interactive exploration of Wolfram's 256 elementary cellular automata. Includes a reusable `Automaton` class and a Jupyter notebook that explains the concepts, shows visualizations, and lets you explore all 256 rules.

## Quick Start

```bash
pip install numpy matplotlib jupyter
jupyter lab notebook/
```

## Structure

```
cellular-automata/
├── automaton/
│   ├── __init__.py
│   └── elementary.py    # Automaton class
├── notebook/
│   └── elementary_ca.ipynb  # Interactive exploration notebook
└── README.md
```

## The Automaton Class

```python
from automaton import ElementaryCA

ca = ElementaryCA(rule=110, width=101, steps=50)
ca.run()
ca.plot()
```

## Links

- [Wolfram's Elementary Cellular Automata](https://mathworld.wolfram.com/ElementaryCellularAutomaton.html)
- [Blog post](https://imadestuff.com/posts/cellular-automata-wolfram/)

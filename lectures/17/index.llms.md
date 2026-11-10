# Heuristic Search

CSI 4106 Introduction to Artificial Intelligence

> **IMPORTANT:**
>
> **Deadline**: Assignment 3 must be submitted no later than November 16, 2026, at 11 PM. Please refer to the assignment description available on Brightspace.
>
> **Deadline**: Quiz 2 is scheduled for Wednesday, November 18, 2026, during class. Additional details are available in the [FAQ](../../course-faq).

## Prepare

- Russell and Norvig ([2020](#ref-Russell:2020aa)), pages 85–92
- Review [Lecture 16](../../lectures/16/index.llms.md), especially state representation, successor generation, the frontier, and breadth-first search.

## Participate

- [slides](slides.llms.md) ([PDF](slides.pdf), [Jupyter Notebook](slides.ipynb))

## Practice

Use the lecture’s Jupyter notebook to:

- Compare misplaced-tile and Manhattan heuristics on the same boards. Compare node expansions and validate solution paths and lengths against BFS on a small sample.
- Set h(s)=0 and explain the relationship between A\*, uniform-cost search, and BFS, including the effect of tie-breaking.

For a larger empirical check, download the [BFS–A\* validation script](validate_astar.py). Run `python3 validate_astar.py 4106 25000` to save results for 25,000 distinct solvable boards; the lecture notebook reads the saved CSV to display histograms and statistics.

## References

Russell, Stuart, and Peter Norvig. 2020. *Artificial Intelligence: A Modern Approach*. 4th ed. Pearson. <http://aima.cs.berkeley.edu/>.

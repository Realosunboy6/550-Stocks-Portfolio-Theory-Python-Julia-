# Meta-RaPS: A Research Synthesis

**Meta-RaPS** (Meta-heuristic for Randomized Priority Search) is a generic, memoryless optimization algorithm for combinatorial problems, originating from work by Gail DePuy, Gary Whitehouse, and collaborators at the University of Central Florida / University of Louisville / Old Dominion University. This document synthesizes the published literature (academic papers, dissertations, ResearchGate abstracts, and industry case studies) into six topic areas: core methodology, theoretical positioning, machine-learning hybridization, scheduling applications, applications beyond scheduling, and benchmarks/open challenges — closing with an assessment of its relevance to combinatorial portfolio-optimization problems, the subject of this repository.

Each section below was researched independently and cites its sources inline and in a trailing `### Sources` list. Where the literature does not directly support a claim, the text says so explicitly rather than extrapolating silently.

## Table of Contents

1. [Core Methodology and Algorithmic Mechanics](#core-methodology-and-algorithmic-mechanics)
2. [Theoretical Positioning and Comparison with Other Metaheuristics](#theoretical-positioning-and-comparison-with-other-metaheuristics)
3. [Hybridization with Machine Learning](#hybridization-with-machine-learning)
4. [Real-World Scheduling Applications](#real-world-scheduling-applications)
5. [Applications Beyond Scheduling](#applications-beyond-scheduling)
6. [Benchmarks, Parameter Tuning, Open Challenges, and Relevance to Portfolio Optimization](#benchmarks-parameter-tuning-open-challenges-and-relevance-to-portfolio-optimization)

---

## Core Methodology and Algorithmic Mechanics

### Origin and Motivation

Meta-RaPS (Meta-heuristic for Randomized Priority Search) grew out of work by Gail W. DePuy and Gary E. Whitehouse at the University of Central Florida around 2001, who built on a modified version of COMSOAL (Computer Method of Sequencing Operations for Assembly Lines) — a classic randomized construction heuristic for assembly-line balancing [DePuy & Whitehouse, 2000](https://www.sciencedirect.com/science/article/abs/pii/S036083520000053X). The approach was formalized and named as a general-purpose metaheuristic by Reinaldo J. Moraga in his 2002 doctoral dissertation, *"Meta-RaPS: An Effective Solution Approach for Combinatorial Problems"* [Moraga, 2002](https://www.researchgate.net/publication/267328739_META-RAPS_UN_ENFOQUE_DE_SOLUCION_EFICAZ_PARA_PROBLEMAS_COMBINATORIOS_META-RAPS_AN_EFFECTIVE_SOLUTION_APPROACH_FOR_COMBINATORIAL_PROBLEMS), which examined parameter setting and applications to the 0-1 Multidimensional Knapsack Problem (MKP) and the Early/Tardy single-machine scheduling problem. DePuy, Moraga, and Whitehouse subsequently demonstrated the method on the Traveling Salesman Problem, reporting that it "outperformed most other solution methodologies in terms of percent difference from optimal" on standard test sets and an industrial truck-routing case study [DePuy, Moraga & Whitehouse, 2005](https://www.sciencedirect.com/science/article/abs/pii/S1366554504000146). Seyhun Hepdogan's 2007 UCF dissertation, *"Meta-RaPS: Parameter Setting and New Applications,"* further extended the method and tackled the parameter-tuning problem directly, since meta-heuristic solution-value distributions are typically non-normal [Hepdogan, 2007](https://stars.library.ucf.edu/etd/914/). The motivation throughout was to design a **generic, high-level, problem-independent strategy** for converting any greedy construction heuristic into a metaheuristic by injecting controlled randomness — explicitly positioned as a simpler, lower-overhead alternative to memory-based or population-based metaheuristics such as Tabu Search, Genetic Algorithms, and Ant Colony Optimization [Moraga, 2002](https://www.researchgate.net/publication/267328739_META-RAPS_UN_ENFOQUE_DE_SOLUCION_EFICAZ_PARA_PROBLEMAS_COMBINATORIOS_META-RAPS_AN_EFFECTIVE_SOLUTION_APPROACH_FOR_COMBINATORIAL_PROBLEMS).

### Two-Phase Structure

Each Meta-RaPS iteration builds and then refines one feasible solution, repeated for a fixed number of iterations *I*:

**Construction phase.** Starting from an empty solution, the algorithm repeatedly selects the next element (e.g., the next job, city, or item) using a *priority rule* native to the problem (shortest processing time, nearest neighbor, greatest density ratio, etc.). Instead of always picking the single best-priority candidate as a pure greedy heuristic would, Meta-RaPS forms a restricted "available list" or candidate list of all feasible elements whose priority value lies within a tolerance of the best value, then samples from that list. This randomized-greedy construction is repeated I times to produce a population of diverse feasible solutions [Hepdogan, 2007](https://stars.library.ucf.edu/etd/914/); [Lan & DePuy, 2006](https://www.sciencedirect.com/science/article/abs/pii/S036083520600101X).

**Improvement phase.** Each constructed solution can optionally be passed through a local-search/neighborhood-search step — commonly swap moves (exchanging the positions of two elements) and insertion moves (relocating one element elsewhere in the sequence) — to push it toward a local optimum before the next construction iteration begins [Lan & DePuy, 2006](https://www.sciencedirect.com/science/article/abs/pii/S036083520600101X); [Cano, García & Pierreval, 2016](https://www.sciencedirect.com/science/article/abs/pii/S1568494616302174). The best solution found across all iterations is reported as the final output.

### Key Parameters and Randomization Control

The classical formulation uses four control parameters [Hepdogan, 2007](https://stars.library.ucf.edu/etd/914/):

- **Number of iterations (I)** — how many times the construction+improvement cycle is run (effectively a random-restart count).
- **Percentage of priority, %p** — the fraction of construction steps in which the single best-priority-value candidate is deterministically chosen; for the remaining (100 − %p) of steps, the next element is drawn at random from the restricted candidate list rather than taken greedily.
- **Percentage of restriction, %r** — defines how "good" an alternative must be to enter the candidate list: any feasible element whose priority value is within %r of the best value qualifies. A larger %r widens the candidate pool (more exploration); %r = 0 collapses construction to pure greedy.
- **Percentage of improvement, %i** — controls how much effort/how many local-search moves are applied during the improvement phase (or, in some formulations, the probability that improvement is invoked at all).

Selection among qualifying candidates is typically **uniform random** within the restricted list (rather than weighted/biased by priority value, distinguishing it from probabilistic mechanisms like ACO's pheromone-weighted choice). This combination — close kin to GRASP's restricted candidate list — gives Meta-RaPS "greater flexibility over COMSOAL and GRASP" by exposing %p, %r, and %i as independently tunable knobs [Hepdogan, 2007](https://stars.library.ucf.edu/etd/914/).

### Why "Memoryless"

Meta-RaPS is classified as a **memoryless metaheuristic**: each iteration constructs a solution from scratch using only the static priority rule and fresh random draws, with no structure carried forward from previous iterations to bias future choices [Lan & DePuy, 2006](https://www.sciencedirect.com/science/article/abs/pii/S036083520600101X). This contrasts with Tabu Search, which maintains a tabu list of recently visited solutions/moves to forbid cycling, and Ant Colony Optimization, which maintains pheromone trails that probabilistically reinforce historically good edges across iterations. The tradeoff is explicit in the literature: memorylessness yields a simpler algorithm with fewer parameters, easier implementation, and natural parallelizability (iterations are independent and can run concurrently), but it forfeits the ability to learn from the search history within a run — each restart "forgets" what worked before. This gap motivated several extensions that graft memory or learning onto the base method, including Path Relinking [Rodriguez, Cano & Pierreval, 2016](https://www.sciencedirect.com/science/article/abs/pii/S1568494616302174), Q-learning, and Estimation of Distribution Algorithms [Ramirez-Lavín et al., 2016](https://www.sciencedirect.com/science/article/abs/pii/S0360835216304077), and explicit memory mechanisms tested on the Set Covering Problem, where memory was found to most improve the construction phase while randomization most improved the improvement phase [Lan & DePuy, 2006](https://www.sciencedirect.com/science/article/abs/pii/S036083520600101X).

### Algorithm Outline (Main Loop)

1. Initialize parameters I, %p, %r, %i; set best solution to null.
2. For each of I iterations:
   a. **Construct**: Begin with an empty/partial solution. While the solution is incomplete: compute priority values for all feasible next elements under the chosen priority rule; with probability %p select the best-priority element deterministically, otherwise build a restricted candidate list of all elements within %r of the best priority value and choose uniformly at random from it; append the chosen element and update feasibility.
   b. **Improve**: Apply local search (swap/insertion moves, controlled by %i) to the completed solution, accepting moves that improve solution quality, until no improving move is found or the improvement budget is exhausted.
   c. **Update**: If this iteration's solution is better than the incumbent best, replace the best solution.
3. After I iterations, return the best solution found.

### Strengths and Limitations

Reported strengths: conceptual and implementation simplicity, very few parameters relative to most metaheuristics, problem-independence (any greedy priority rule can be "Meta-RaPS-ified"), natural diversification while preserving the quality signal of the priority rule, and strong empirical performance on TSP, MKP, set covering, scheduling, and vehicle-routing benchmarks, often competitive with or superior to more complex methods [DePuy, Moraga & Whitehouse, 2005](https://www.sciencedirect.com/science/article/abs/pii/S1366554504000146); [Lan & DePuy, 2006](https://www.sciencedirect.com/science/article/abs/pii/S036083520600101X). Reported limitations: as a purely memoryless, multi-start design it cannot learn across iterations, so solution quality depends heavily on the chosen priority rule and on manual tuning of %p/%r/%i, which several follow-up studies addressed via non-parametric genetic-algorithm-based parameter setting [Hepdogan, 2007](https://stars.library.ucf.edu/etd/914/) and via hybridization with memory/learning mechanisms (Path Relinking, Q-learning, EDAs) to close the gap with adaptive metaheuristics [Rodriguez, Cano & Pierreval, 2016](https://www.sciencedirect.com/science/article/abs/pii/S1568494616302174); [Ramirez-Lavín et al., 2016](https://www.sciencedirect.com/science/article/abs/pii/S0360835216304077).

### Sources

- [DePuy & Whitehouse, 2000, "Applying the COMSOAL computer heuristic to the constrained resource allocation problem"](https://www.sciencedirect.com/science/article/abs/pii/S036083520000053X)
- [Moraga, 2002, "META-RaPS: An Effective Solution Approach for Combinatorial Problems"](https://www.researchgate.net/publication/267328739_META-RAPS_UN_ENFOQUE_DE_SOLUCION_EFICAZ_PARA_PROBLEMAS_COMBINATORIOS_META-RAPS_AN_EFFECTIVE_SOLUTION_APPROACH_FOR_COMBINATORIAL_PROBLEMS)
- [DePuy, Moraga & Whitehouse, 2005, "Meta-RaPS: a simple and effective approach for solving the traveling salesman problem," Transportation Research Part E 41(2):115-130](https://www.sciencedirect.com/science/article/abs/pii/S1366554504000146)
- [Lan & DePuy, 2006, "On the effectiveness of incorporating randomness and memory into a multi-start metaheuristic with application to the Set Covering Problem"](https://www.sciencedirect.com/science/article/abs/pii/S036083520600101X)
- [Hepdogan, 2007, "Meta-RaPS: Parameter Setting and New Applications," UCF dissertation](https://stars.library.ucf.edu/etd/914/)
- [Rodriguez, Cano & Pierreval, 2016, "Local search versus Path Relinking in metaheuristics: Redesigning Meta-RaPS with application to the multidimensional knapsack problem"](https://www.sciencedirect.com/science/article/abs/pii/S1568494616302174)
- [Ramirez-Lavín et al., 2016, "Integrating estimation of distribution algorithms versus Q-learning into Meta-RaPS for solving the 0-1 multidimensional knapsack problem"](https://www.sciencedirect.com/science/article/abs/pii/S0360835216304077)
- [ResearchGate, "Metaheuristic for Randomized Priority Search (Meta-RaPS): A Tutorial"](https://www.researchgate.net/publication/299509758_Metaheuristic_for_Randomized_Priority_Search_Meta-RaPS_A_Tutorial)

---

## Theoretical Positioning and Comparison with Other Metaheuristics

### Position in the Metaheuristic Taxonomy

Meta-RaPS belongs to the family of **construction-heuristic-plus-local-search hybrids** rather than to the population-based or trajectory-based families that dominate the metaheuristics literature. At each iteration it (1) builds a feasible solution greedily using a priority rule, but randomizes the rule's choice, and then (2) applies an improvement/local-search phase to the constructed solution [ResearchGate Tutorial](https://www.researchgate.net/publication/299509758_Metaheuristic_for_Randomized_Priority_Search_Meta-RaPS_A_Tutorial). This two-phase structure places Meta-RaPS in the same conceptual lineage as **GRASP** (Greedy Randomized Adaptive Search Procedure) [Wikipedia](https://en.wikipedia.org/wiki/Greedy_randomized_adaptive_search_procedure), distinguishing it from:

- **Population-based methods** (GA, PSO, EDAs) — evolve a population via selection/crossover/mutation rather than repeatedly reconstructing single solutions from scratch.
- **Trajectory-based local search methods** (Simulated Annealing, Tabu Search) — perturb and move through a single solution's neighborhood over time, using probabilistic acceptance (SA) or memory structures (TS) to escape local optima.
- **Swarm/nature-inspired methods** (ACO) — use distributed agents and pheromone-like reinforcement to bias construction collectively.

Meta-RaPS instead injects diversification at the *construction* stage and relies on restarts/iterations rather than a memory structure or population to maintain diversity [Semantic Scholar / DePuy-Whitehouse COMSOAL work](https://www.semanticscholar.org/paper/Applying-the-COMSOAL-computer-heuristic-to-the-DePuy-Whitehouse/2d94cf39a6283a0d576c8c5b55d5b38fc07ef7e2).

### Head-to-Head Performance Comparisons

- **TSP:** Meta-RaPS "outperformed most other solution methodologies in terms of percent difference from optimal" on standard test instances, and an industrial case study reported over 50% reduction in engineering time and more than $2.5 million in annual transportation savings versus the firm's existing approach [DePuy/Whitehouse, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1366554504000146).
- **0-1 MKP:** Benchmarked against published results from other heuristics, competitive on solution quality while solving the largest benchmark instances "in significantly less time" [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S036083520400138X). Later work redesigned the local-search phase using **Path Relinking** [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1568494616302174), and a separate study hybridizing with **EDA** and **Q-learning** found "Meta-RaPS EDA performs better than Meta-RaPS Q, and both are superior to the original Meta-RaPS and existing benchmark data" [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0360835216304077) — implying the base algorithm leaves headroom that learning-based hybridization can close.
- **Set Covering Problem (SCP):** A Meta-RaPS-derived heuristic was tested on 80 OR-Library SCP instances and reported as "one of two known SCP heuristics to find all optimal/best known solutions" on that set [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0377221705008313).
- **Scheduling problems** (blocking flow shops, aerial refueling scheduling, project selection/scheduling): Meta-RaPS has been benchmarked in NASA-funded aerial refueling scheduling work [NASA NTRS](https://ntrs.nasa.gov/api/citations/20110012105/downloads/20110012105.pdf) and flow-shop/project-scheduling contexts [ResearchGate](https://www.researchgate.net/publication/283015771_Scheduling_Blocking_Flow_Shops_Using_Meta-RaPS), [Emerald/Kybernetes](https://www.emerald.com/k/article-abstract/43/9-10/1483/267577/A-metaheuristic-algorithm-for-project-selection?redirectedFrom=fulltext), positioning it as a credible alternative to GA- and TS-based scheduling metaheuristics.

### Computational Complexity and Runtime Behavior

Secondary sources summarizing the tutorial state that "run times for Meta-RaPS [are] not significantly affected by the size of the problem," attributed to its reliance on a fast greedy construction step repeated over iterations rather than an expensive neighborhood-exhaustive search [ResearchGate Tutorial](https://www.researchgate.net/publication/299509758_Metaheuristic_for_Randomized_Priority_Search_Meta-RaPS_A_Tutorial). No source gives a formal asymptotic complexity bound — the runtime claims are empirical/benchmark-based rather than theoretical.

### Robustness to Parameter Settings

Meta-RaPS is consistently marketed as simpler to configure than GA (crossover/mutation rates, population size) or SA (cooling schedule, initial temperature): nominally only iteration count plus %priority and %restriction (sometimes %improvement) [NASA NTRS](https://ntrs.nasa.gov/api/citations/20110012105/downloads/20110012105.pdf). This claim is qualified rather than absolute: the literature acknowledges performance "depends on the fine tuning of two main parameters," motivating a dedicated UCF doctoral thesis on parameter setting [Hepdogan, UCF STARS](https://stars.library.ucf.edu/etd/914/) and a data-mining-based hybridization using a decision-tree model for online, adaptive tuning [Al-Duoli & Rabadi, ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1877050914013465).

### Criticisms and Limitations

- **Parameter sensitivity despite the "few parameters" claim** — multiple dedicated tuning papers/theses are themselves evidence that naive fixed settings are not robust across problem classes.
- **Dependence on priority-rule quality** — solution-quality ceiling is bounded by how informative the chosen greedy rule is.
- **Local-search phase as a bottleneck** — researchers replaced it with Path Relinking and layered EDA/Q-learning on top to outperform the base algorithm, implicitly critiquing the plain improvement phase.
- **General metaheuristic risks** — premature convergence and degraded scalability on very large, high-dimensional instances are flagged as common failure modes for randomized-construction/local-search methods generally [Artificial Intelligence Review survey](https://link.springer.com/article/10.1007/s10462-025-11377-6).

### Synthesis

The literature positions Meta-RaPS as favorable specifically for **combinatorial problems that already have a strong, well-understood greedy/priority construction rule** (sequencing, knapsack, covering, scheduling), where runtime predictability and ease of implementation matter as much as squeezing out the last fraction of optimality. It appears weaker when no good priority rule exists, when the local-search phase becomes the bottleneck on harder instances, or when provable convergence/optimality guarantees are required.

### Sources

- [Metaheuristic for Randomized Priority Search (Meta-RaPS): A Tutorial — ResearchGate](https://www.researchgate.net/publication/299509758_Metaheuristic_for_Randomized_Priority_Search_Meta-RaPS_A_Tutorial)
- [Greedy Randomized Adaptive Search Procedure — Wikipedia](https://en.wikipedia.org/wiki/Greedy_randomized_adaptive_search_procedure)
- [Applying the COMSOAL computer heuristic to the constrained resource allocation problem — Semantic Scholar](https://www.semanticscholar.org/paper/Applying-the-COMSOAL-computer-heuristic-to-the-DePuy-Whitehouse/2d94cf39a6283a0d576c8c5b55d5b38fc07ef7e2)
- [Meta-RaPS: a simple and effective approach for solving the TSP — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1366554504000146)
- [Meta-RaPS TSP paper — ResearchGate mirror](https://www.researchgate.net/publication/222242560_Meta-RaPS_A_simple_and_effective_approach_for_solving_the_traveling_salesman_problem)
- [Meta-RaPS approach for the 0-1 Multidimensional Knapsack Problem — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S036083520400138X)
- [Local search versus Path Relinking in metaheuristics: Redesigning Meta-RaPS — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1568494616302174)
- [Integrating EDA versus Q-learning into Meta-RaPS for the 0-1 MKP — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0360835216304077)
- [An effective and simple heuristic for the set covering problem — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0377221705008313)
- [Scheduling Blocking Flow Shops Using Meta-RaPS — ResearchGate](https://www.researchgate.net/publication/283015771_Scheduling_Blocking_Flow_Shops_Using_Meta-RaPS)
- [A metaheuristic algorithm for project selection and scheduling — Emerald/Kybernetes](https://www.emerald.com/k/article-abstract/43/9-10/1483/267577/A-metaheuristic-algorithm-for-project-selection?redirectedFrom=fulltext)
- [Meta-RaPS Algorithm for the Aerial Refueling Scheduling Problem — NASA NTRS](https://ntrs.nasa.gov/api/citations/20110012105/downloads/20110012105.pdf)
- ["Meta-RaPS: Parameter Setting and New Applications" — Hepdogan, UCF STARS](https://stars.library.ucf.edu/etd/914/)
- [Data Mining based Hybridization of Meta-RaPS — ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1877050914013465)
- [Data Mining based Hybridization of Meta-RaPS — Digital Commons ODU author copy](https://digitalcommons.odu.edu/emse_fac_pubs/9/)
- [Applications, classifications, and challenges: a comprehensive evaluation of recently developed metaheuristics — Artificial Intelligence Review](https://link.springer.com/article/10.1007/s10462-025-11377-6)

---

## Hybridization with Machine Learning

### Overview: a small but real direct literature

Unlike pure parameter-tuning studies, a specific and traceable line of work hybridizes Meta-RaPS directly with machine learning, centered on researchers at Old Dominion University (Ghaith Rabadi's group) and earlier parameter-setting work at the University of Central Florida.

### Documented Meta-RaPS + ML attempts

**Decision trees and association rules for on-line parameter tuning.** Fatemah Al-Duoli's PhD dissertation, *"Meta-RaPS Hybridization with Machine Learning Algorithms"* [Al-Duoli, 2015](https://digitalcommons.odu.edu/emse_etds/40/), integrates a decision tree (supervised learning) to perform on-line tuning of Meta-RaPS's construction parameters, mining favorable parameter settings from the knowledge gained in earlier iterations. Association rules (unsupervised learning) were also examined. The hybrid was tested on Vehicle Routing Problem benchmark instances. Companion papers: [Al-Duoli & Rabadi, "Data Mining Based Hybridization of Meta-RaPS"](https://digitalcommons.odu.edu/emse_fac_pubs/9/) ([ResearchGate copy](https://www.researchgate.net/publication/275541549_Data_Mining_based_Hybridization_of_Meta-RaPS)), and [IEEE Xplore](https://ieeexplore.ieee.org/document/8488390/).

**Learning-based intensification (precursor work).** [Al-Duoli & Rabadi, "Employing Learning to Improve the Performance of Meta-RaPS," Procedia Computer Science 20 (2013), pp. 46–51](https://www.sciencedirect.com/science/article/pii/S1877050913010375) frames the work within the Adaptive Memory Programming paradigm — using learning to give the inherently *memoryless* Meta-RaPS a memory mechanism via an Inductive Decision Tree (IDT).

**Q-learning and Estimation of Distribution Algorithms.** [Arin & Rabadi, "Integrating Estimation of Distribution Algorithms versus Q-Learning into Meta-RaPS for Solving the 0-1 Multidimensional Knapsack Problem," Computers & Industrial Engineering, Vol. 112 (2017), pp. 706–720](https://www.sciencedirect.com/science/article/abs/pii/S0360835216304077) embeds Q-learning directly inside Meta-RaPS's construction phase, and separately tests a stochastic EDA variant. Result: Meta-RaPS-EDA outperformed Meta-RaPS-Q-Learning, but both improved on the original Meta-RaPS and on other published 0-1 MKP benchmarks.

**Path Relinking as a learned intensification mechanism.** [Arin & Rabadi, "Performance of an Intensification Strategy Based on Learning in a Metaheuristic: Meta-RaPS with Path Relinking" (Springer, 2016)](https://link.springer.com/chapter/10.1007/978-3-319-26024-2_6) and the companion [Applied Soft Computing study](https://www.sciencedirect.com/science/article/abs/pii/S1568494616302174) learn "good" attributes shared by elite solutions and steer construction toward them.

### Analogy to GRASP-family learning (explicit caveat: not Meta-RaPS-specific)

Because Meta-RaPS is structurally a close cousin of GRASP, the broader GRASP+ML literature is informative by analogy, though none of these papers mention Meta-RaPS by name: **Reactive GRASP** self-adjusts its restrictiveness parameter based on solution quality found so far [ResearchGate figure](https://www.researchgate.net/figure/Traditional-GRASP-Reactive-GRASP-and-GRASP-learning-Processing-Time_fig7_221909672); **"GRASP-learning"** uses a Q-learning coefficient to bias candidate selection [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-642-10701-6_14); modern **neural construction heuristics** (pointer networks, GNNs) represent the deep-learning-era version of the same idea — see [awesome-ml4co](https://github.com/Thinklab-SJTU/awesome-ml4co) and [DeepACO](https://arxiv.org/pdf/2309.14032).

### Specific ML techniques paired with this algorithm class

- **Decision trees / association rules** — parameter and rule prediction from historical run data.
- **Reinforcement learning (Q-learning)** — adaptive, in-loop control of construction-phase choices.
- **Estimation of Distribution Algorithms** — a probabilistic learning model of the solution distribution.
- **Path Relinking as learned memory** — a learned intensification operator drawing on elite-solution attributes.
- **Surrogate fitness models** (Kriging, RBF, SVM) — established in the broader surrogate-assisted metaheuristics literature, but **no Meta-RaPS-specific surrogate-model paper was found**; flagged as a speculative extension by analogy ([survey, Springer 2024](https://link.springer.com/article/10.1007/s40747-024-01465-5); [MDPI overview](https://www.mdpi.com/2076-3417/15/16/9068)).

### Benefits and challenges reported

**Benefits:** Meta-RaPS-EDA and Meta-RaPS-Q-Learning both exceeded plain Meta-RaPS and competing benchmarks on 0-1 MKP; Path-Relinking-based learning "outperformed other approaches used in the literature" on the same problem; decision-tree-based tuning reduces the manual burden of setting %priority/%restriction.

**Challenges:** EDA's probabilistic model and Q-learning's state-action tables add computational/implementation overhead relative to memoryless Meta-RaPS; the broader ML4CO literature flags generalization across problem sizes and sample-efficiency as open issues, which plausibly extends to any Meta-RaPS+ML hybrid (inference, not direct citation).

### Open gaps noted by authors

Hepdogan's dissertation [UCF STARS](https://stars.library.ucf.edu/etd/914/) explicitly states that dynamic/intelligent parameter setting for Meta-RaPS "has not been extensively studied in the literature." The Al-Duoli/Rabadi and Arin/Rabadi papers each test only a single problem domain (VRP; 0-1 MKP), leaving cross-domain generalization of any specific ML hybridization as an implicit, unaddressed gap. No source discusses neural-network-based priority-rule learning or surrogate-fitness modeling specifically for Meta-RaPS.

### Sources

- [Al-Duoli, F. (2015), "Meta-RaPS Hybridization with Machine Learning Algorithms" (PhD dissertation, ODU)](https://digitalcommons.odu.edu/emse_etds/40/)
- [Al-Duoli, F. & Rabadi, G., "Data Mining Based Hybridization of Meta-RaPS" (ODU Faculty Publications)](https://digitalcommons.odu.edu/emse_fac_pubs/9/)
- [Al-Duoli & Rabadi, "Data Mining based Hybridization of Meta-RaPS" (ResearchGate)](https://www.researchgate.net/publication/275541549_Data_Mining_based_Hybridization_of_Meta-RaPS)
- [Al-Duoli & Rabadi, "Hybridizing Meta-RaPS with Machine Learning Algorithms" (IEEE Xplore, 2018)](https://ieeexplore.ieee.org/document/8488390/)
- [Al-Duoli & Rabadi (2013), "Employing Learning to Improve the Performance of Meta-RaPS," Procedia Computer Science 20, 46–51](https://www.sciencedirect.com/science/article/pii/S1877050913010375)
- [Arin & Rabadi (2017), "Integrating EDA versus Q-Learning into Meta-RaPS for the 0-1 MKP," Computers & Industrial Engineering 112, 706–720](https://www.sciencedirect.com/science/article/abs/pii/S0360835216304077)
- [Arin & Rabadi, same paper (ResearchGate copy)](https://www.researchgate.net/publication/309518096_Integrating_Estimation_of_Distribution_Algorithms_versus_Q-Learning_into_Meta-RaPS_for_Solving_the_0-1_Multidimensional_Knapsack_Problem)
- [Arin & Rabadi (2016), "Performance of an Intensification Strategy Based on Learning in a Metaheuristic: Meta-RaPS with Path Relinking" (Springer)](https://link.springer.com/chapter/10.1007/978-3-319-26024-2_6)
- [Arin & Rabadi, "Local search versus Path Relinking in metaheuristics: Redesigning Meta-RaPS..." (Applied Soft Computing)](https://www.sciencedirect.com/science/article/abs/pii/S1568494616302174)
- [Hepdogan, S. (2006), "Meta-RaPS: Parameter Setting and New Applications" (PhD dissertation, UCF)](https://stars.library.ucf.edu/etd/914/)
- [Traditional GRASP, Reactive GRASP and GRASP-learning comparison figure (ResearchGate)](https://www.researchgate.net/figure/Traditional-GRASP-Reactive-GRASP-and-GRASP-learning-Processing-Time_fig7_221909672)
- [Parallel hybrid GA + GRASP + reinforcement learning for TSP (Springer)](https://link.springer.com/chapter/10.1007/978-3-642-10701-6_14)
- [awesome-ml4co (GitHub curated list)](https://github.com/Thinklab-SJTU/awesome-ml4co)
- [DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization (arXiv)](https://arxiv.org/pdf/2309.14032)
- [Surrogate-assisted evolutionary algorithms for expensive combinatorial optimization: a survey (Springer, 2024)](https://link.springer.com/article/10.1007/s40747-024-01465-5)
- [Single-Objective Surrogate Models for Continuous Metaheuristics: An Overview (MDPI)](https://www.mdpi.com/2076-3417/15/16/9068)

---

## Real-World Scheduling Applications

### Resource-Constrained Project Scheduling (RCPSP)

The RCPSP is the application most closely tied to Meta-RaPS's origins. The problem requires sequencing project activities subject to precedence relations and limited renewable resources to minimize project makespan — an NP-hard combinatorial problem central to construction, engineering, and large-program management. Meta-RaPS traces directly back to [DePuy & Whitehouse, 2001, "A simple and effective heuristic for the resource constrained project scheduling problem"](https://www.tandfonline.com/doi/abs/10.1080/00207540110060608), which adapted the Modified COMSOAL heuristic to RCPSP. The construction phase applies a priority rule (e.g., minimum slack, latest finish time) but randomizes selection among top-ranked eligible activities rather than always picking the single best one — generating a different feasible schedule on each pass and keeping the best result across iterations, the direct ancestor of the formal Meta-RaPS framework later codified by [Moraga, 2002](https://www.researchgate.net/publication/299509758_Metaheuristic_for_Randomized_Priority_Search_Meta-RaPS_A_Tutorial). DePuy and Whitehouse benchmarked against the Patterson set and PSPLIB instances (J30/J60/J120 families), the reference datasets cited across follow-on RCPSP heuristic comparisons. A related precursor paper is [DePuy & Whitehouse, "Applying the COMSOAL computer heuristic to the constrained resource allocation problem"](https://www.semanticscholar.org/paper/Applying-the-COMSOAL-computer-heuristic-to-the-DePuy-Whitehouse/2d94cf39a6283a0d576c8c5b55d5b38fc07ef7e2).

### Flow Shop Scheduling

Meta-RaPS has been applied to blocking flow shop scheduling, where jobs pass through a sequence of machines with no buffer between stages — a constraint common in chemical processing and steel/metal production lines. In ["Scheduling Blocking Flow Shops Using Meta-RaPS"](https://www.researchgate.net/publication/283015771_Scheduling_Blocking_Flow_Shops_Using_Meta-RaPS) ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1877050915030410)), the objective is minimizing makespan, with construction using the NEH (Nawaz-Enscore-Ham) insertion heuristic as its base rule, randomized by Meta-RaPS. Benchmarked against leading construction heuristics and metaheuristics on Taillard flow shop instances, with competitive results.

### Parallel Machine and Tardiness-Based Scheduling

Meta-RaPS has been applied to unrelated parallel machine scheduling with sequence- and machine-dependent setup times [Journal of Intelligent Manufacturing](https://link.springer.com/article/10.1007/s10845-005-5514-0), finding all optimal solutions on small instances and outperforming an existing benchmark heuristic on larger instances. A related military application is the Aerial Refueling Scheduling Problem (ARSP) — scheduling fighter aircraft across tanker aircraft to minimize total weighted tardiness [NASA NTRS](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20110012105.pdf) ([academia.edu mirror](https://www.academia.edu/91220635/Meta_RaPS_Algorithm_for_the_Aerial_Refueling_Scheduling_Problem)), using a Meta-RaPS variant built on the Apparent Tardiness Cost (ATC) priority rule to produce near-optimal schedules quickly.

### Adjacent Logistics/Routing Deployment

A documented industry deployment of Meta-RaPS (applied to the TSP) is informative for the broader scheduling narrative: a national U.S. trucking company used an automated Meta-RaPS-based tool to generate less-than-truckload pickup-and-delivery routes, reporting more than a 50% reduction in route-engineering time and over $2.5 million in annual transportation cost savings [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1366554504000146). No comparably documented Meta-RaPS deployment for personnel/crew or workforce scheduling was found — an apparent application gap relative to RCPSP, flow shop, and parallel-machine scheduling.

### Common Threads

Scheduling problems suit Meta-RaPS's construction-plus-improvement design because: (1) classical scheduling heuristics are already priority-rule-based, giving Meta-RaPS a ready-made deterministic backbone to randomize; (2) feasibility is cheap to maintain incrementally as activities/jobs are inserted one at a time; (3) the restart/perturbation structure naturally escapes the local optima that pure greedy dispatching rules get stuck in; and (4) the algorithm's simplicity makes it attractive for industry deployment where engineering time and explainability matter as much as optimality gap.

### Sources

- [DePuy & Whitehouse (2001), original Meta-RaPS-precursor RCPSP heuristic paper](https://www.tandfonline.com/doi/abs/10.1080/00207540110060608)
- [DePuy & Whitehouse, COMSOAL applied to constrained resource allocation (precursor work)](https://www.semanticscholar.org/paper/Applying-the-COMSOAL-computer-heuristic-to-the-DePuy-Whitehouse/2d94cf39a6283a0d576c8c5b55d5b38fc07ef7e2)
- [Moraga/Whitehouse/DePuy Meta-RaPS tutorial](https://www.researchgate.net/publication/299509758_Metaheuristic_for_Randomized_Priority_Search_Meta-RaPS_A_Tutorial)
- [Blocking flow shop scheduling with Meta-RaPS + NEH construction, Taillard benchmark](https://www.researchgate.net/publication/283015771_Scheduling_Blocking_Flow_Shops_Using_Meta-RaPS)
- [ScienceDirect version of the blocking flow shop Meta-RaPS paper](https://www.sciencedirect.com/science/article/pii/S1877050915030410)
- [Unrelated parallel machine scheduling with setup times](https://link.springer.com/article/10.1007/s10845-005-5514-0)
- [NASA technical report, Meta-RaPS-ATC for Aerial Refueling Scheduling Problem](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20110012105.pdf)
- [Mirror of the aerial refueling Meta-RaPS paper](https://www.academia.edu/91220635/Meta_RaPS_Algorithm_for_the_Aerial_Refueling_Scheduling_Problem)
- [Meta-RaPS for TSP, including the LTL trucking industry case study](https://www.sciencedirect.com/science/article/abs/pii/S1366554504000146)

---

## Applications Beyond Scheduling

Although Meta-RaPS was originally developed and is most extensively documented in production scheduling contexts, its developers explicitly designed it as a general-purpose combinatorial metaheuristic, and a distinct stream of papers tests it against classic, non-scheduling NP-hard problems. The evidence base is real but concentrated in a handful of problem classes; some commonly expected domains (VRP, bin packing, assembly line balancing, finance) returned no documented direct application.

### Traveling Salesman Problem (TSP)

[DePuy, Moraga & Whitehouse, 2005](https://ideas.repec.org/a/eee/transe/v41y2005i2p115-130.html) apply the standard construction-plus-improvement framework to TSP: a nearest-neighbor-style priority rule ranks candidate next-city edges, randomized between top-priority choice and a restricted "good" set, followed by local-search improvement. Results show Meta-RaPS outperforming most competing methods in percent deviation from optimal [ResearchGate mirror](https://www.researchgate.net/publication/222242560_Meta-RaPS_A_simple_and_effective_approach_for_solving_the_traveling_salesman_problem). The paper also reports an industry case study: an automated Meta-RaPS-based TSP tool embedded in a truck-route assignment model, estimated at over 50% reduction in engineering time and more than $2.5 million in annual savings [TRID record](https://trid.trb.org/View/748727). No documented Meta-RaPS application to the classic vehicle routing problem (VRP) was found — VRP appears to be an open/unaddressed extension of the TSP work.

### Knapsack and Resource Allocation Problems

The 0-1 Multidimensional Knapsack Problem (MKP) is the second major non-scheduling domain. The original application [Arntzen, DePuy & Whitehouse](https://www.sciencedirect.com/science/article/abs/pii/S036083520400138X) ([ACM record](https://dl.acm.org/doi/10.5555/1081562.1651688)) implements four greedy priority rules based on pseudo-utility ratios, with two local-search techniques in the improvement phase, reported to outperform many competing MKP heuristics. Later extended via EDA/Q-learning hybridization [ScienceDirect, 2016](https://www.sciencedirect.com/science/article/abs/pii/S0360835216304077) and a Path Relinking redesign [IEEE Xplore](https://ieeexplore.ieee.org/document/6335159/). Closely related: an IEEE paper applies Meta-RaPS to project-activity resource allocation under precedence/resource constraints [IEEE Xplore](https://ieeexplore.ieee.org/document/1049428/), tested on RCPSP benchmark sets.

### Set Covering (Facility/Network-Type Problems)

[Lan, DePuy & Whitehouse, 2007, "An effective and simple heuristic for the set covering problem," EJOR 176(3), 1387–1403](https://ideas.repec.org/a/eee/ejores/v176y2007i3p1387-1403.html) adds randomized priority-rule selection, a penalty mechanism for "worst columns," and a core-problem reduction step. Tested on 80 OR-Library SCP instances, including non-unicost instances up to 1000 rows × 10,000 columns and unicost instances up to 28,160 rows × 11,264 columns. No direct application to facility location, p-median, or maximal covering location problems specifically was found beyond this generic set-covering work.

### Manufacturing/Operations Problems Beyond Scheduling

No Meta-RaPS papers were found for bin packing or assembly line balancing — both adjacent to COMSOAL (Meta-RaPS's ancestor heuristic) in background material, but with no Meta-RaPS-specific results located.

### Finance, Portfolio Selection, and Asset Allocation

No direct Meta-RaPS application to portfolio optimization, asset allocation, or finance was found in any search variation. This appears to be a genuine gap rather than a search artifact. The closest adjacent literature is the broader field of metaheuristics for cardinality-constrained portfolio selection, which shares deep structural similarity to the MKP problems Meta-RaPS has solved (both are resource/cardinality-constrained 0-1 selection problems): e.g., [time-limited metaheuristics for cardinality-constrained portfolio optimisation (arXiv, 2023)](https://arxiv.org/pdf/2307.04045) and [practical portfolio optimization with metaheuristics under pre-assignment and margin-trading constraints (arXiv, 2025)](https://arxiv.org/pdf/2503.15965). These confirm the problem family is metaheuristic-amenable in the same way as MKP, but use other methods (genetic algorithms, particle swarm, etc.), not Meta-RaPS — an unexploited transfer opportunity rather than documented prior work.

### Breadth of Validation

Meta-RaPS has been empirically validated on a moderately broad but specific set of classical combinatorial problems beyond scheduling: TSP (with a real industry routing deployment), 0-1 MKP (with multiple algorithmic enhancements over a decade-plus research program), set covering, and resource-constrained project scheduling/resource allocation. All share Meta-RaPS's core architecture — priority-rule-driven, randomized greedy construction followed by local-search improvement — generalizing naturally to any problem expressible as sequential, rankable item/edge/column selection under constraints. Domains requiring fundamentally different solution representations (continuous allocation as in portfolio weights, multi-vehicle routing, two-dimensional packing) remain largely or entirely untested in the published Meta-RaPS literature as of this search.

### Sources

- [DePuy, Moraga & Whitehouse (2005), Meta-RaPS for TSP, journal record](https://ideas.repec.org/a/eee/transe/v41y2005i2p115-130.html)
- [RG abstract page for the same TSP paper](https://www.researchgate.net/publication/222242560_Meta-RaPS_A_simple_and_effective_approach_for_solving_the_traveling_salesman_problem)
- [TRID record summarizing TSP paper including the truck-routing industry case study](https://trid.trb.org/View/748727)
- [Meta-RaPS approach for the 0-1 Multidimensional Knapsack Problem (original MKP paper)](https://www.sciencedirect.com/science/article/abs/pii/S036083520400138X)
- [ACM/Computers & Industrial Engineering record of the Meta-RaPS MKP paper](https://dl.acm.org/doi/10.5555/1081562.1651688)
- [Meta-RaPS with Path Relinking for the 0-1 MKP (IEEE)](https://ieeexplore.ieee.org/document/6335159/)
- [Integrating EDA vs. Q-learning into Meta-RaPS for the 0-1 MKP](https://www.sciencedirect.com/science/article/abs/pii/S0360835216304077)
- [Local search vs. Path Relinking redesign of Meta-RaPS for MKP](https://www.sciencedirect.com/science/article/abs/pii/S1568494616302174)
- [Meta-RaPS approach for solving the resource allocation problem (RCPSP), IEEE](https://ieeexplore.ieee.org/document/1049428/)
- [Lan, DePuy & Whitehouse (2007), heuristic for the Set Covering Problem, EJOR record](https://ideas.repec.org/a/eee/ejores/v176y2007i3p1387-1403.html)
- [RG abstract page for the same SCP paper](https://www.researchgate.net/publication/4939132_An_Effective_and_Simple_Heuristic_for_the_Set_Covering_Problem)
- [ScienceDirect record of the SCP paper](https://www.sciencedirect.com/science/article/abs/pii/S0377221705008313)
- [Time-limited metaheuristics for cardinality-constrained portfolio optimisation (adjacent, non-Meta-RaPS finance literature)](https://arxiv.org/pdf/2307.04045)
- [Practical portfolio optimization with metaheuristics: pre-assignment constraint and margin trading (adjacent, non-Meta-RaPS finance literature)](https://arxiv.org/pdf/2503.15965)
- [Meta-RaPS tutorial overview referencing the range of combinatorial applications](https://www.researchgate.net/publication/299509758_Metaheuristic_for_Randomized_Priority_Search_Meta-RaPS_A_Tutorial)

---

## Benchmarks, Parameter Tuning, Open Challenges, and Relevance to Portfolio Optimization

### Parameter Tuning Methodology

Meta-RaPS is governed by a small set of control parameters — **%priority** (greediness, the fraction of construction steps taking the locally best choice), **%restriction** (how far below the best priority value a feasible candidate may fall and still be eligible for random selection), and **%improvement** (controls acceptance during the local-search/restart phase), together with the **number of iterations/restarts** [Rabadi et al., Meta-RaPS Tutorial](https://www.researchgate.net/publication/299509758_Metaheuristic_for_Randomized_Priority_Search_Meta-RaPS_A_Tutorial). The literature's dominant tuning approach is a **full factorial design of experiments (DOE)**: parameters are tested at a small number of discrete levels, all combinations are run, and ANOVA or mean-response comparison identifies significant factors and interactions. A representative tuned configuration from a NASA-documented aerial refueling scheduling study: iterations = 10,000, priority = 25%, restriction = 60%, improvement = 70% [NASA NTRS](https://ntrs.nasa.gov/api/citations/20110012105/downloads/20110012105.pdf). This DOE-then-ANOVA pattern recurs across Meta-RaPS papers, though exact factor levels are re-tuned per problem domain rather than transferred wholesale — an important caveat for adapting the method to a new domain such as portfolio selection.

### Benchmark Instances and Reported Performance

- **0-1 MKP:** OR-Library-style instances (n = 100, 250, 500; m = 5, 10, 30; 30 instances per combination), compared against known optimal/best-known values, generally competitive or superior in percent-deviation-from-optimal [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S036083520400138X).
- **TSP:** evaluated on established TSP test sets by percent difference from optimal tour length, extended to the industry truck-routing case study (>50% reduction in engineering time, >$2.5M/year savings) [ResearchGate](https://www.researchgate.net/publication/222242560_Meta-RaPS_A_simple_and_effective_approach_for_solving_the_traveling_salesman_problem).
- **RCPSP / parallel machine / aerial refueling scheduling:** benchmarked against exact optimal solutions for small instances (up to ~12 jobs) and other metaheuristics (e.g., simulated annealing) for larger instances (up to ~60 jobs); somewhat worse deviation-from-optimal on small instances but substantially better CPU time, with relative quality advantage growing with instance size [NASA NTRS](https://ntrs.nasa.gov/api/citations/20110012105/downloads/20110012105.pdf).

A note on evidence quality: the RCPSP literature broadly uses **PSPLIB** (J30/J60/J90/J120 instance sets) as the field-standard benchmark [Kolisch & Sprecher, PSPLIB](https://www.sciencedirect.com/science/article/abs/pii/S0377221796001701), but direct confirmation that the specific Meta-RaPS scheduling papers used PSPLIB (versus custom or Patterson-set instances) could not be verified — treated as plausible but unconfirmed.

### Open Challenges Noted in the Literature

- **Domain-specific re-tuning burden.** No universal parameter set; a new problem class requires a fresh DOE tuning campaign.
- **Solution-quality/time trade-off at small scale.** Meta-RaPS can be measurably worse than exact or other metaheuristic methods on small instances, with its advantage concentrated at larger sizes.
- **No theoretical convergence guarantees.** Like most metaheuristics, empirical performance is reported without formal convergence proofs or worst-case bounds — a general limitation of the class rather than a Meta-RaPS-specific finding in the sources reviewed.
- **Limited multi-objective and dynamic/online extensions.** Retrieved sources describe single-objective, static, offline formulations only; no multi-objective or online/dynamic Meta-RaPS variant was found, suggesting this remains an open, largely undiscussed extension.

### Bridge to Portfolio Optimization

Cardinality-constrained (discrete) portfolio selection — choosing exactly K assets out of N, possibly with minimum/maximum buy-in thresholds — converts the classical convex Markowitz mean-variance quadratic program into a **mixed-integer quadratic program that is NP-hard** [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S2210650218309702). Structurally this is close kin to the knapsack and scheduling problems Meta-RaPS targets: in MKP, a subset of items is selected under capacity constraints to maximize value; in cardinality-constrained portfolio selection, a subset of assets is selected under a count/weight constraint to maximize risk-adjusted return. Both are subset-selection-plus-continuous-allocation problems with combinatorial explosion in the discrete part.

A Meta-RaPS-style heuristic adapted to this setting would plausibly follow the same two-phase template documented for knapsack and scheduling: (1) a **construction phase** using a greedy priority rule — e.g., a reward-to-risk ratio per asset, analogous to a return/weight density measure in knapsack — where most of the time the next asset added is the best-ranked feasible candidate, but with %priority probability a random choice is made among candidates within %restriction of the best; and (2) an **improvement phase** that perturbs the selected asset subset (swap one held asset for an unheld one, akin to a 2-opt move in TSP) and re-solves the continuous weight allocation as a smaller, exactly-solvable convex QP over the chosen K assets, accepting improving or near-improving moves across many random restarts. This description is a **reasonable inference by analogy** to the documented Meta-RaPS framework — it has not been verified as an approach explicitly published under the Meta-RaPS name for portfolio selection. Existing portfolio metaheuristic literature instead favors genetic algorithms, particle swarm optimization, firefly algorithms, and other population-based methods for the cardinality-constrained Markowitz problem [Soft Computing](https://link.springer.com/article/10.1007/s00500-023-08177-x); [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4060745/), so a direct precedent for "Meta-RaPS applied to portfolios" was not found and should not be claimed as established practice.

Given Meta-RaPS's documented profile — simple to implement, computationally cheap per iteration, competitive on larger combinatorial instances but requiring problem-specific parameter re-tuning and offering no optimality guarantee — it is a *plausible* candidate worth empirical testing for cardinality-constrained portfolio construction (especially for large universes where exact MIQP solvers become slow), but the case rests on structural analogy to knapsack/scheduling rather than on direct published evidence within Meta-RaPS literature itself.

### Sources

- [Rabadi, G. et al., "Metaheuristic for Randomized Priority Search (Meta-RaPS): A Tutorial," ResearchGate](https://www.researchgate.net/publication/299509758_Metaheuristic_for_Randomized_Priority_Search_Meta-RaPS_A_Tutorial)
- [NASA Technical Reports Server, "Meta-RaPS Algorithm for the Aerial Refueling Scheduling Problem"](https://ntrs.nasa.gov/api/citations/20110012105/downloads/20110012105.pdf)
- [Lucas, Rabadi & Mollaghasemi, "Meta-RaPS approach for the 0-1 Multidimensional Knapsack Problem," ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S036083520400138X)
- ["Integrating estimation of distribution algorithms versus Q-learning into Meta-RaPS," ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0360835216304077)
- ["Local search versus Path Relinking in metaheuristics: Redesigning Meta-RaPS," ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1568494616302174)
- [DePuy, Whitehouse et al., "Meta-RaPS: A simple and effective approach for solving the TSP," ResearchGate](https://www.researchgate.net/publication/222242560_Meta-RaPS_A_simple_and_effective_approach_for_solving_the_traveling_salesman_problem)
- [Kolisch & Sprecher, "PSPLIB - A project scheduling problem library," ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0377221796001701)
- ["An efficient hybrid metaheuristic algorithm for cardinality constrained portfolio optimization," ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S2210650218309702)
- ["Meta-heuristics for portfolio optimization," Soft Computing / Springer](https://link.springer.com/article/10.1007/s00500-023-08177-x)
- ["Firefly Algorithm for Cardinality Constrained Mean-Variance Portfolio Optimization Problem with Entropy Diversity Constraint," PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4060745/)

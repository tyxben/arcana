"""
Arcana: Context Management Benchmark

Demonstrates Arcana's token efficiency vs naive full-context approach.
No API key required -- runs entirely with simulated conversations.

Results show how Arcana's WorkingSetBuilder compresses multi-turn history
while preserving coherence:
- Keeps system prompt + 6 recent messages verbatim
- Scores middle messages by role, keyword overlap, error content
- Relevance-aware compression: important messages keep more detail
- Transparent ContextDecision audit for every compression

Quick demo -- how context compression works:

    from arcana.context.builder import WorkingSetBuilder, estimate_tokens
    from arcana.contracts.context import TokenBudget
    from arcana.contracts.llm import Message, MessageRole

    builder = WorkingSetBuilder(
        identity="You are a research assistant.",
        token_budget=TokenBudget(total_window=8000, response_reserve=2000),
        goal="Help the user understand quantum computing",
    )
    # ... accumulate 30 turns of messages ...
    compressed = builder.build_conversation_context(messages, turn=30)
    naive_tokens = sum(estimate_tokens(m.content or "") for m in messages)
    arcana_tokens = sum(estimate_tokens(m.content or "") for m in compressed)
    savings = (1 - arcana_tokens / naive_tokens) * 100
    print(f"{len(messages)} messages -> {len(compressed)} messages, {savings:.0f}% token savings")
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from arcana.context.builder import WorkingSetBuilder, estimate_tokens
from arcana.contracts.context import TokenBudget
from arcana.contracts.llm import Message, MessageRole

# ---------------------------------------------------------------------------
# Simulated conversation: a research assistant helping with quantum computing
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are Quanta, an advanced research assistant specializing in quantum computing, "
    "quantum information theory, and their applications. You explain complex topics "
    "clearly, cite relevant papers when possible, and help the user build intuition. "
    "When the user asks about implementation, provide pseudocode or real code examples. "
    "Always clarify assumptions and note when a topic is still an active area of research."
)

# Conversation turns: (user_message, assistant_response)
# Designed to simulate realistic message lengths and variety.
CONVERSATION_TURNS: list[tuple[str, str]] = [
    # Turn 1 -- short intro
    (
        "What is quantum computing and why should I care about it?",
        "Quantum computing uses quantum-mechanical phenomena -- superposition and "
        "entanglement -- to process information in fundamentally different ways than "
        "classical computers. While a classical bit is either 0 or 1, a qubit can be "
        "in a superposition of both states simultaneously. This enables quantum computers "
        "to explore many solutions in parallel for certain problems. You should care "
        "because quantum computers promise exponential speedups for specific tasks like "
        "factoring large numbers (Shor's algorithm), simulating molecules for drug "
        "discovery, and solving optimization problems that are intractable classically.",
    ),
    # Turn 2 -- deeper question
    (
        "Can you explain superposition more formally?",
        "In quantum mechanics, a qubit's state is described by a state vector in a "
        "two-dimensional complex Hilbert space. We write it as |psi> = alpha|0> + beta|1>, "
        "where alpha and beta are complex amplitudes satisfying |alpha|^2 + |beta|^2 = 1. "
        "The key insight is that before measurement, the qubit genuinely exists in both "
        "states at once -- this is not just classical uncertainty. When we measure, the "
        "state 'collapses' to |0> with probability |alpha|^2 or |1> with probability "
        "|beta|^2. For n qubits, the state space is 2^n dimensional, which is why quantum "
        "computers can represent exponentially more information than classical ones. "
        "The Bloch sphere provides a beautiful geometric visualization: any pure qubit "
        "state maps to a point on the surface of a unit sphere.",
    ),
    # Turn 3 -- entanglement
    (
        "How does entanglement work? Give me the math.",
        "Entanglement is a correlation between qubits that has no classical analogue. "
        "Consider two qubits. A separable (non-entangled) state can always be written as "
        "a tensor product: |psi> = |a> tensor |b>. An entangled state CANNOT be decomposed "
        "this way. The canonical example is the Bell state: |phi+> = (1/sqrt(2))(|00> + |11>). "
        "If you measure the first qubit and get |0>, the second qubit is INSTANTLY |0> as "
        "well, regardless of distance. Einstein called this 'spooky action at a distance.' "
        "Mathematically, entanglement is characterized by the von Neumann entropy of the "
        "reduced density matrix: S(rho_A) = -Tr(rho_A log rho_A). For a maximally entangled "
        "state of two qubits, S = ln(2). Bell's theorem (1964) proves these correlations "
        "cannot be explained by any local hidden variable theory, confirmed experimentally "
        "by Aspect et al. (1982) and many subsequent tests. The 2022 Nobel Prize in Physics "
        "was awarded to Aspect, Clauser, and Zeilinger for this work.",
    ),
    # Turn 4 -- practical question
    (
        "What quantum gates do I need to know?",
        "The essential single-qubit gates are: (1) Pauli gates X, Y, Z -- X is the quantum "
        "NOT gate, Z flips the phase, Y combines both. (2) Hadamard gate H -- creates "
        "superposition: H|0> = (|0>+|1>)/sqrt(2). (3) Phase gates S and T -- add phase "
        "rotations of pi/2 and pi/4 respectively. (4) Rotation gates Rx, Ry, Rz -- "
        "parameterized rotations around the Bloch sphere axes. For multi-qubit operations: "
        "(1) CNOT (controlled-NOT) -- the workhorse of entanglement, flips target qubit "
        "when control is |1>. (2) CZ (controlled-Z) -- applies Z to target when control "
        "is |1>. (3) Toffoli (CCNOT) -- universal for classical computation. (4) SWAP -- "
        "exchanges two qubits. These gates form a universal gate set: any unitary operation "
        "can be approximated to arbitrary precision using {H, T, CNOT}. This is proven "
        "by the Solovay-Kitaev theorem. In practice, hardware native gates vary: "
        "superconducting qubits often use sqrt(X) and CZ, trapped ions use Molmer-Sorensen.",
    ),
    # Turn 5 -- error correction
    (
        "Why is quantum error correction so important?",
        "Quantum systems are extraordinarily fragile. Decoherence -- interaction with the "
        "environment -- destroys quantum information on timescales of microseconds to "
        "milliseconds for current hardware. Error rates per gate are around 0.1-1% for "
        "state-of-the-art systems (Google Sycamore: ~0.5% two-qubit error). Without error "
        "correction, errors accumulate exponentially with circuit depth. Quantum error "
        "correction (QEC) encodes a logical qubit into many physical qubits. The surface "
        "code is the leading candidate: it arranges qubits in a 2D grid, tolerates errors "
        "up to ~1% per gate, and requires only nearest-neighbor interactions. The catch: "
        "you need roughly 1000-10000 physical qubits per logical qubit depending on the "
        "desired error rate. Google's recent results with their 72-qubit Bristlecone chip "
        "and IBM's 1121-qubit Condor processor are steps toward this threshold, but we "
        "are still in the NISQ (Noisy Intermediate-Scale Quantum) era. Active research "
        "areas include topological codes, concatenated codes, and bosonic codes.",
    ),
    # Turn 6 -- algorithm question
    (
        "Explain Shor's algorithm at a high level.",
        "Shor's algorithm factors integers in polynomial time, threatening RSA encryption. "
        "The key insight is reducing factoring to period-finding, then using quantum "
        "Fourier transform (QFT) to find the period efficiently. Here's the high-level "
        "flow: (1) Pick a random a < N. (2) Check gcd(a, N) -- if > 1, we found a factor. "
        "(3) Find the period r of f(x) = a^x mod N using QPE (quantum phase estimation). "
        "(4) If r is even and a^(r/2) != -1 mod N, then gcd(a^(r/2) +/- 1, N) gives "
        "factors. The quantum speedup is in step 3: classically, finding r requires "
        "exponential time, but QFT does it in O((log N)^2 log log N) with O(log N) qubits. "
        "For RSA-2048, we would need roughly 4000 logical qubits, or about 20 million "
        "physical qubits with current error correction overhead. Estimate: 8 hours of "
        "quantum computation. This is why post-quantum cryptography (lattice-based, "
        "hash-based) is being standardized now by NIST -- the threat is real but not "
        "yet practical.",
    ),
    # Turn 7 -- implementation
    (
        "Show me a simple quantum circuit in Qiskit.",
        "Here's a Bell state circuit -- the 'hello world' of quantum computing:\n\n"
        "```python\n"
        "from qiskit import QuantumCircuit, transpile\n"
        "from qiskit_aer import AerSimulator\n\n"
        "# Create circuit with 2 qubits and 2 classical bits\n"
        "qc = QuantumCircuit(2, 2)\n\n"
        "# Apply Hadamard to qubit 0 (creates superposition)\n"
        "qc.h(0)\n\n"
        "# CNOT with control=0, target=1 (creates entanglement)\n"
        "qc.cx(0, 1)\n\n"
        "# Measure both qubits\n"
        "qc.measure([0, 1], [0, 1])\n\n"
        "# Simulate\n"
        "sim = AerSimulator()\n"
        "compiled = transpile(qc, sim)\n"
        "result = sim.run(compiled, shots=1024).result()\n"
        "counts = result.get_counts()\n"
        "print(counts)  # {'00': ~512, '11': ~512}\n"
        "```\n\n"
        "You'll always see |00> or |11> -- never |01> or |10>. That's entanglement "
        "in action. The two qubits are perfectly correlated even though each individual "
        "measurement is random. Try adding qc.x(0) before the Hadamard to get the "
        "|phi-> state, or qc.x(1) after the CNOT for |psi+>.",
    ),
    # Turn 8 -- tool-like response (simulating a search result)
    (
        "What are the latest breakthroughs in quantum computing this year?",
        "Based on recent publications and announcements: (1) Google achieved quantum error "
        "correction below the surface code threshold using their 72-qubit processor, "
        "demonstrating that adding more qubits actually reduces errors -- a historic "
        "milestone. (2) IBM launched their Heron processor with 133 fixed-frequency "
        "transmon qubits and significantly improved two-qubit gate fidelity (99.5%). "
        "(3) Microsoft announced a topological qubit breakthrough using Majorana fermions "
        "in indium arsenide nanowires, though peer review is ongoing. (4) QuEra and "
        "Harvard demonstrated a 48-logical-qubit system using neutral atoms with "
        "reconfigurable connectivity. (5) China's Jiuzhang 3.0 photonic system claimed "
        "quantum advantage in Gaussian boson sampling with 255 detected photons. "
        "(6) Post-quantum cryptography standards (CRYSTALS-Kyber, CRYSTALS-Dilithium) "
        "were finalized by NIST and are being deployed in major browsers and TLS libraries.",
    ),
    # Turn 9 -- comparison question
    (
        "How do superconducting qubits compare to trapped ion qubits?",
        "This is one of the most important questions in the field. Let me compare:\n\n"
        "SUPERCONDUCTING (Google, IBM, Rigetti):\n"
        "- Gate speed: very fast (~20-100 ns for single, ~200 ns for two-qubit)\n"
        "- Coherence: moderate (50-100 us for T1, improving)\n"
        "- Connectivity: nearest-neighbor on 2D grid (requires SWAP overhead)\n"
        "- Scalability: strong fab compatibility with semiconductor processes\n"
        "- Gate fidelity: 99.5-99.9% single, 99.0-99.5% two-qubit\n"
        "- Drawback: requires dilution refrigerators at 15 mK\n\n"
        "TRAPPED IONS (IonQ, Quantinuum, Alpine Quantum):\n"
        "- Gate speed: slower (~1-10 us single, ~100 us two-qubit)\n"
        "- Coherence: excellent (seconds to minutes!)\n"
        "- Connectivity: all-to-all (no SWAP overhead)\n"
        "- Scalability: challenging beyond ~50 ions in one trap\n"
        "- Gate fidelity: 99.99% single, 99.5-99.9% two-qubit (best in class)\n"
        "- Drawback: slower clock speed limits circuit throughput\n\n"
        "My assessment: trapped ions currently win on fidelity and coherence, "
        "superconducting wins on speed and scalability. The race is far from over -- "
        "neutral atoms (Pasqal, QuEra) are emerging as a dark horse with good scaling "
        "properties and reconfigurable connectivity.",
    ),
    # Turn 10 -- application question
    (
        "What real-world problems can quantum computers solve today?",
        "Honestly? Very few, with genuine quantum advantage. Let me be clear about what's "
        "real vs. hype:\n\n"
        "DEMONSTRATED (but not yet practical advantage):\n"
        "- Random circuit sampling (Google, 2019) -- proves quantum computers work\n"
        "- Boson sampling (Jiuzhang) -- similar proof of concept\n"
        "- Small molecule simulation (H2, LiH, BeH2) -- correct but classically easy\n\n"
        "NEAR-TERM PROMISING (3-5 years):\n"
        "- Variational quantum eigensolvers (VQE) for chemistry\n"
        "- Quantum approximate optimization (QAOA) for logistics\n"
        "- Quantum machine learning on structured data\n"
        "- Quantum sensing and metrology (already useful!)\n\n"
        "LONG-TERM REVOLUTIONARY (10+ years, requires fault tolerance):\n"
        "- Drug discovery: simulating protein folding and molecular interactions\n"
        "- Materials science: designing room-temperature superconductors\n"
        "- Cryptanalysis: breaking RSA/ECC (Shor's algorithm)\n"
        "- Optimization: supply chain, financial modeling, ML training\n\n"
        "The honest truth: we're in the 'transistor era' of quantum computing. "
        "The technology works but isn't yet economically useful. Most quantum advantage "
        "claims should be viewed skeptically. The real breakthroughs will come when "
        "we cross the fault-tolerance threshold.",
    ),
    # Turn 11 -- follow-up
    (
        "Tell me more about VQE for chemistry.",
        "Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical algorithm "
        "designed for near-term quantum hardware. The goal: find the ground state energy "
        "of a molecular Hamiltonian. Here's how it works:\n\n"
        "1. ENCODE: Map the molecular Hamiltonian to a qubit Hamiltonian using "
        "Jordan-Wigner or Bravyi-Kitaev transformation.\n"
        "2. PREPARE: Create a parameterized quantum circuit (ansatz) that prepares "
        "trial wavefunctions. Common ansatze: UCCSD (unitary coupled cluster), "
        "hardware-efficient ansatz, ADAPT-VQE.\n"
        "3. MEASURE: Estimate <psi(theta)|H|psi(theta)> by measuring Pauli terms.\n"
        "4. OPTIMIZE: Use a classical optimizer (COBYLA, L-BFGS-B, SPSA) to minimize "
        "the energy by adjusting circuit parameters theta.\n"
        "5. REPEAT: Iterate until convergence.\n\n"
        "The challenge: for useful molecules (like FeMoCo in nitrogen fixation, ~54 "
        "electrons), you need ~100+ qubits with low enough error rates. Current "
        "demonstrations are limited to molecules with ~10-20 electrons. Paper to read: "
        "Peruzzo et al., Nature Communications 2014 (first VQE experiment).",
    ),
    # Turn 12 -- debugging scenario
    (
        "I tried running a VQE on IBM Quantum but got weird results. The energy is way too high.",
        "This is a common issue! Several things could be going wrong:\n\n"
        "ERROR: High VQE energy on real hardware\n\n"
        "LIKELY CAUSES:\n"
        "1. NOISE: Gate errors accumulate. Try error mitigation: zero-noise extrapolation "
        "(ZNE) or probabilistic error cancellation (PEC). Qiskit has `qiskit_aer.noise` "
        "for noise model simulation.\n\n"
        "2. BARREN PLATEAUS: If using a hardware-efficient ansatz with random initialization, "
        "gradients vanish exponentially with qubit count. Fix: use a chemistry-aware ansatz "
        "(UCCSD) or ADAPT-VQE, and initialize near the Hartree-Fock state.\n\n"
        "3. MEASUREMENT SHOTS: Too few shots = noisy expectation values. Use at least "
        "8192 shots, or better, use measurement grouping to reduce the number of circuits.\n\n"
        "4. OPTIMIZER CHOICE: COBYLA tends to get stuck in local minima. Try SPSA "
        "(designed for noisy objectives) or gradient-free methods.\n\n"
        "5. TRANSPILATION: Check that your circuit depth after transpilation isn't too "
        "deep. Use `transpile(qc, backend, optimization_level=3)` and check the depth.\n\n"
        "Debug steps: (a) Run the same circuit on a simulator first. (b) Compare with "
        "exact diagonalization for small systems. (c) Plot energy vs iteration to check "
        "convergence. (d) Monitor circuit depth after transpilation.",
    ),
    # Turn 13 -- brief exchange
    (
        "Got it, I'll try SPSA. Thanks!",
        "Good choice! SPSA (Simultaneous Perturbation Stochastic Approximation) only "
        "needs 2 function evaluations per iteration regardless of the number of "
        "parameters, making it efficient for noisy quantum hardware. Set the learning "
        "rate schedule carefully -- start with a = 0.1, c = 0.1 and decay. Let me know "
        "how it goes!",
    ),
    # Turn 14 -- topic shift
    (
        "Switching topics -- what is quantum machine learning?",
        "Quantum machine learning (QML) sits at the intersection of quantum computing and "
        "ML. There are several paradigms:\n\n"
        "1. QUANTUM KERNEL METHODS: Encode classical data into quantum states, then "
        "compute kernel functions (inner products) in quantum Hilbert space. Theoretical "
        "advantage: can compute kernels that are classically intractable. Practical "
        "issue: the advantage often disappears when you account for encoding overhead.\n\n"
        "2. VARIATIONAL QUANTUM CIRCUITS (VQC): Parameterized quantum circuits used as "
        "ML models. Think of them as quantum neural networks. Training uses a classical "
        "optimizer to tune gate parameters. Current research: avoiding barren plateaus, "
        "designing good ansatze, understanding expressibility.\n\n"
        "3. QUANTUM-ENHANCED CLASSICAL ML: Use quantum computers to speed up specific "
        "subroutines -- e.g., HHL algorithm for linear systems (but with caveats), "
        "quantum PCA, Grover-enhanced search.\n\n"
        "4. CLASSICAL ML FOR QUANTUM: Using classical neural networks to help quantum "
        "tasks -- error decoding, pulse optimization, ansatz design.\n\n"
        "My honest take: QML is overhyped in the near term. Most claimed advantages "
        "don't hold under scrutiny (see: Tang's dequantization results). Long-term, "
        "the most promising direction is quantum simulation for generating training "
        "data that classical computers can't produce.",
    ),
    # Turn 15 -- specific question
    (
        "How do you encode classical data into qubits?",
        "Data encoding (also called data embedding or feature maps) is crucial and often "
        "the bottleneck. Main approaches:\n\n"
        "AMPLITUDE ENCODING: Encode N-dimensional data vector into log2(N) qubits by "
        "mapping components to amplitudes. Pro: exponential compression. Con: state "
        "preparation requires O(N) gates, negating the qubit savings.\n\n"
        "ANGLE ENCODING: Map each feature to a rotation angle on a separate qubit. "
        "x_i -> Ry(x_i)|0>. Pro: simple, O(n) qubits for n features. Con: no "
        "compression, limited expressiveness.\n\n"
        "IQP ENCODING: Use parameterized circuits with ZZ entangling gates: "
        "exp(i*x_i*x_j*ZZ). Creates complex feature maps in Hilbert space. Used in "
        "quantum kernel methods.\n\n"
        "HAMILTONIAN ENCODING: Evolve under a Hamiltonian parameterized by data. "
        "Natural for time-series and physics-inspired data.\n\n"
        "The encoding choice often matters more than the quantum model architecture. "
        "There's a theorem by Schuld (2021) showing that many quantum models are "
        "essentially Fourier series with frequencies determined by the encoding.",
    ),
    # Turn 16 -- code request
    (
        "Show me a simple quantum classifier in PennyLane.",
        "Here's a minimal quantum binary classifier:\n\n"
        "```python\n"
        "import pennylane as qml\n"
        "import numpy as np\n"
        "from pennylane import numpy as pnp\n\n"
        "dev = qml.device('default.qubit', wires=2)\n\n"
        "@qml.qnode(dev)\n"
        "def circuit(weights, x):\n"
        "    # Data encoding\n"
        "    qml.AngleEmbedding(x, wires=range(2))\n"
        "    # Variational layers\n"
        "    qml.StronglyEntanglingLayers(weights, wires=range(2))\n"
        "    # Measurement\n"
        "    return qml.expval(qml.PauliZ(0))\n\n"
        "# Initialize\n"
        "n_layers = 3\n"
        "weights = pnp.random.randn(n_layers, 2, 3, requires_grad=True)\n\n"
        "# Training data (XOR problem)\n"
        "X = pnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], requires_grad=False)\n"
        "Y = pnp.array([1, -1, -1, 1], requires_grad=False)\n\n"
        "# Optimize\n"
        "opt = qml.GradientDescentOptimizer(stepsize=0.4)\n"
        "for epoch in range(100):\n"
        "    for x, y in zip(X, Y):\n"
        "        def cost(w):\n"
        "            return (circuit(w, x) - y) ** 2\n"
        "        weights = opt.step(cost, weights)\n\n"
        "# Test\n"
        "for x, y in zip(X, Y):\n"
        "    pred = circuit(weights, x)\n"
        "    print(f'x={x}, target={y:+d}, pred={pred:+.3f}')\n"
        "```\n\n"
        "This solves XOR -- a non-linearly separable problem -- using quantum "
        "entanglement to create non-linear decision boundaries. In practice, you'd "
        "use qml.optimize.AdamOptimizer and batch training for real datasets.",
    ),
    # Turn 17 -- clarification
    (
        "What's the difference between NISQ and fault-tolerant quantum computing?",
        "NISQ (Noisy Intermediate-Scale Quantum) and fault-tolerant quantum computing "
        "represent two eras:\n\n"
        "NISQ ERA (NOW - ~2030?):\n"
        "- Qubits: 50-1000+ physical qubits\n"
        "- Error rates: 0.1-1% per gate\n"
        "- No error correction (or limited)\n"
        "- Circuit depth: limited to ~100-1000 gates before noise dominates\n"
        "- Algorithms: variational (VQE, QAOA), sampling, analog simulation\n"
        "- Status: can do things classical computers can't, but not yet useful things\n\n"
        "FAULT-TOLERANT ERA (2030+?):\n"
        "- Qubits: millions of physical qubits, thousands of logical qubits\n"
        "- Error rates: effectively zero (through QEC)\n"
        "- Full error correction: surface code or better\n"
        "- Circuit depth: unlimited (in principle)\n"
        "- Algorithms: Shor, Grover, HHL, quantum simulation at arbitrary precision\n"
        "- Status: the promised land, but physically daunting\n\n"
        "The gap between these eras is often called the 'quantum valley of death' -- "
        "we can build NISQ machines but they're not yet useful enough to justify their "
        "cost. The key milestone: demonstrating that adding more qubits to an error-"
        "corrected system reduces the logical error rate (Google achieved this in 2024).",
    ),
    # Turn 18 -- short exchange
    (
        "When will we have a useful quantum computer?",
        "My best estimate: 5-7 years for narrow quantum advantage in chemistry/materials, "
        "10-15 years for cryptographically relevant quantum computers. But predictions in "
        "this field have a terrible track record -- Feynman proposed quantum computing in "
        "1982, and we're still not there. The bottleneck isn't physics, it's engineering.",
    ),
    # Turn 19 -- deep dive request
    (
        "Let's go deeper on the surface code. How does it actually work?",
        "The surface code is elegant. Imagine a 2D grid of qubits with two types:\n\n"
        "DATA QUBITS: arranged on edges of a lattice, store quantum information.\n"
        "MEASURE QUBITS: on vertices and faces, detect errors without reading data.\n\n"
        "The key operators:\n"
        "- X stabilizers: products of Pauli X on qubits around each face\n"
        "- Z stabilizers: products of Pauli Z on qubits around each vertex\n\n"
        "Error detection works like this:\n"
        "1. Repeatedly measure all stabilizers (syndrome extraction).\n"
        "2. Errors show up as -1 eigenvalues on nearby stabilizers.\n"
        "3. A classical decoder (minimum weight perfect matching, MWPM) identifies "
        "the most likely error pattern.\n"
        "4. Apply corrections.\n\n"
        "The beautiful part: you never need to know the actual quantum state! "
        "Stabilizer measurements only reveal error syndromes, preserving the encoded "
        "information. The logical qubit is encoded in the topology -- a non-trivial "
        "loop of X operators across the lattice represents the logical X, and a "
        "non-trivial loop of Z operators represents logical Z.\n\n"
        "Code distance d (lattice side length) determines protection: can correct "
        "any error affecting fewer than d/2 qubits. For d=17 and physical error rate "
        "0.1%, the logical error rate drops to ~10^-12 per round. That's 2*17^2 - 1 "
        "= 577 physical qubits per logical qubit at this distance.\n\n"
        "Active research: faster decoders (MWPM is O(n^3), neural decoders are faster), "
        "magic state distillation for non-Clifford gates, code concatenation strategies.",
    ),
    # Turn 20 -- practical question
    (
        "What programming frameworks should I learn for quantum computing?",
        "Here's my ranked recommendation:\n\n"
        "TIER 1 (Learn these first):\n"
        "- Qiskit (IBM) -- most widely used, great tutorials, access to real hardware. "
        "Now modular: qiskit-terra (circuits), qiskit-aer (simulation), qiskit-nature "
        "(chemistry), qiskit-ml (machine learning).\n"
        "- Cirq (Google) -- more research-focused, great for custom experiments. "
        "Integrates with TensorFlow Quantum.\n\n"
        "TIER 2 (Specialize with these):\n"
        "- PennyLane (Xanadu) -- best for quantum ML, differentiable programming, "
        "hardware-agnostic. Can run on any backend.\n"
        "- Amazon Braket -- unified API for multiple hardware vendors (IonQ, Rigetti, "
        "OQC). Good if you want to compare platforms.\n\n"
        "TIER 3 (Niche but important):\n"
        "- Stim (Google) -- specialized for Clifford circuits and error correction "
        "simulation. Blazingly fast.\n"
        "- OpenFermion (Google) -- molecular simulation and fermionic Hamiltonians.\n"
        "- TKET (Quantinuum) -- advanced circuit optimization compiler.\n\n"
        "Start with Qiskit, build intuition, then branch into PennyLane for ML or "
        "Cirq for research. All are free and open-source with Python APIs.",
    ),
    # Turn 21 -- error scenario
    (
        "I'm getting an error: 'CircuitError: duplicate qubit argument' in Qiskit. Help!",
        "This error means you're using the same qubit twice in a single gate operation. "
        "Common causes:\n\n"
        "ERROR: CircuitError: duplicate qubit argument\n\n"
        "FIX 1 -- CNOT self-reference:\n"
        "  # WRONG: qc.cx(0, 0)  -- can't use same qubit as control and target\n"
        "  # RIGHT: qc.cx(0, 1)\n\n"
        "FIX 2 -- Loop variable collision:\n"
        "  # WRONG:\n"
        "  for i in range(n):\n"
        "      qc.cx(i, i)  # Always same qubit!\n"
        "  # RIGHT:\n"
        "  for i in range(n-1):\n"
        "      qc.cx(i, i+1)\n\n"
        "FIX 3 -- Register indexing:\n"
        "  # If using named registers, make sure qr1[0] and qr2[0] aren't the same\n"
        "  # physical qubit. Check with: print(qc.qubits)\n\n"
        "Debug: add `print(qc.draw())` before the failing line to visualize the "
        "circuit state. Also check `qc.num_qubits` to ensure you have enough qubits.",
    ),
    # Turn 22 -- brief
    (
        "Fixed it, was a loop bug. Thanks!",
        "Classic off-by-one! Quantum circuits are unforgiving with qubit indices. "
        "A good practice: always draw the circuit with `qc.draw('mpl')` after "
        "construction to visually verify the structure.",
    ),
    # Turn 23 -- theoretical question
    (
        "What is quantum supremacy/advantage and has it been achieved?",
        "The terminology has evolved:\n\n"
        "QUANTUM SUPREMACY (deprecated term, now 'quantum computational advantage'):\n"
        "Performing a computation that no classical computer can do in reasonable time.\n\n"
        "KEY CLAIMS:\n"
        "1. Google (2019): 53-qubit Sycamore processor sampled random circuits in 200 "
        "seconds. Claimed classical equivalent: 10,000 years on Summit supercomputer. "
        "IBM disputed this, showing a classical simulation in 2.5 days. Still, the "
        "quantum computer used 3M times less energy.\n\n"
        "2. USTC/Pan Jianwei (2020-2023): Jiuzhang photonic systems demonstrated "
        "advantage in Gaussian boson sampling. Jiuzhang 3.0 used 255 detected photons.\n\n"
        "3. Xanadu (2022): Programmable photonic processor showed advantage in a "
        "specific sampling task.\n\n"
        "THE CATCH: All demonstrations are for artificial problems designed to be hard "
        "for classical computers. No practical quantum advantage exists yet -- meaning "
        "no quantum computer has solved a real-world problem faster than classical "
        "alternatives. This remains the holy grail.\n\n"
        "My view: quantum advantage has been achieved in a narrow technical sense, "
        "but practical quantum advantage remains years away. The goalposts keep moving "
        "as classical algorithms and hardware improve.",
    ),
    # Turn 24 -- application
    (
        "How could quantum computing help with climate change?",
        "Several promising directions:\n\n"
        "1. CATALYST DESIGN: Simulating the nitrogen fixation process (Haber-Bosch uses "
        "2% of world energy). Quantum computers could model the FeMoCo active site of "
        "nitrogenase enzyme to design better catalysts. Requires ~100 logical qubits.\n\n"
        "2. BATTERY MATERIALS: Simulating lithium-ion and solid-state battery chemistry "
        "to find better materials. Understanding ion transport at the quantum level could "
        "lead to breakthrough energy storage.\n\n"
        "3. CARBON CAPTURE: Modeling molecular interactions between CO2 and potential "
        "capture materials (metal-organic frameworks, zeolites) to design more efficient "
        "carbon capture systems.\n\n"
        "4. SOLAR CELLS: Quantum simulation of exciton dynamics in photovoltaic materials, "
        "potentially discovering more efficient light-harvesting configurations.\n\n"
        "5. GRID OPTIMIZATION: Quantum optimization algorithms for power grid management, "
        "renewable energy scheduling, and smart grid routing.\n\n"
        "6. WEATHER/CLIMATE MODELING: Long-term -- quantum-enhanced differential equation "
        "solvers could improve climate model resolution.\n\n"
        "Timeline reality check: Most of these applications require fault-tolerant "
        "quantum computers. Near-term NISQ devices might help with simplified models "
        "of battery chemistry or small catalyst molecules.",
    ),
    # Turn 25 -- summary request
    (
        "Can you summarize the key takeaways from our entire conversation?",
        "Here's a distillation of our quantum computing deep dive:\n\n"
        "FUNDAMENTALS:\n"
        "- Qubits use superposition (alpha|0> + beta|1>) and entanglement for computation\n"
        "- Universal gate set: {H, T, CNOT} can approximate any quantum operation\n"
        "- Measurement collapses quantum states -- information extraction is probabilistic\n\n"
        "CURRENT STATE:\n"
        "- We're in the NISQ era: 50-1000+ noisy qubits, limited circuit depth\n"
        "- Quantum advantage demonstrated for artificial tasks, not practical ones\n"
        "- Error correction works but needs 1000x qubit overhead\n"
        "- Major platforms: superconducting (Google, IBM), trapped ions (IonQ, Quantinuum)\n\n"
        "ALGORITHMS:\n"
        "- Shor's (factoring), Grover's (search), VQE (chemistry), QAOA (optimization)\n"
        "- Quantum ML: promising but overhyped, encoding is the bottleneck\n\n"
        "PRACTICAL PATH:\n"
        "- Learn Qiskit first, then PennyLane/Cirq\n"
        "- Near-term: chemistry simulation, optimization\n"
        "- Long-term: cryptography, drug discovery, materials science, climate tech\n"
        "- Fault tolerance: 5-10 years for first useful demonstrations\n\n"
        "The field is at an inflection point -- the science works, the engineering "
        "challenge remains. It's like the internet in 1993: clearly transformative, "
        "but the killer apps haven't arrived yet.",
    ),
    # Turn 26 -- new sub-topic
    (
        "What about quantum networking and the quantum internet?",
        "Quantum networking is about distributing entanglement over long distances. "
        "The vision: a 'quantum internet' that enables:\n\n"
        "1. QUANTUM KEY DISTRIBUTION (QKD): Already commercial (ID Quantique, Toshiba). "
        "Uses quantum mechanics to guarantee eavesdropper detection. BB84 protocol is "
        "the foundation. China has a 2000km QKD backbone (Beijing-Shanghai).\n\n"
        "2. DISTRIBUTED QUANTUM COMPUTING: Link multiple small quantum processors into "
        "one large virtual machine. Requires quantum teleportation of states.\n\n"
        "3. QUANTUM SENSING NETWORKS: Synchronized atomic clocks, gravitational wave "
        "detection, dark matter searches.\n\n"
        "CHALLENGES:\n"
        "- Photon loss in fiber: quantum signals can't be amplified (no-cloning theorem)\n"
        "- Quantum repeaters: needed for long distances, require quantum memory + "
        "entanglement swapping. Still early-stage technology.\n"
        "- Rate: current QKD systems achieve ~1 Mbit/s over 50km, dropping exponentially\n\n"
        "TIMELINE: Metropolitan QKD networks: NOW. City-to-city via satellite: 5 years. "
        "Full quantum internet with repeaters: 10-15 years. The first quantum repeater "
        "demonstrations are happening in labs this year.",
    ),
    # Turn 27 -- philosophical
    (
        "Does quantum computing change our understanding of reality?",
        "Profoundly, yes. Quantum computing is essentially applied philosophy of physics:\n\n"
        "1. COMPUTATIONAL PERSPECTIVE: The Extended Church-Turing thesis -- that any "
        "physical process can be efficiently simulated by a classical computer -- is "
        "almost certainly wrong. Quantum computers suggest that nature computes with "
        "quantum mechanics, and simulating it classically is inherently expensive.\n\n"
        "2. INFORMATION IS PHYSICAL: Landauer's principle says erasing a bit generates "
        "heat (kT ln 2). Quantum computing adds: information is not just physical, it's "
        "quantum physical. The universe stores and processes information in qubits.\n\n"
        "3. MANY WORLDS: Deutsch (a founder of quantum computing) argues that quantum "
        "computation is evidence for the Many Worlds interpretation -- where do the "
        "exponential computations 'happen' if not in parallel branches of reality?\n\n"
        "4. COMPLEXITY AND PHYSICS: The relationship between BQP (what quantum computers "
        "can solve efficiently) and NP tells us something deep about the structure of "
        "the universe. If BQP != NP (likely), even quantum mechanics has computational "
        "limits.\n\n"
        "5. HOLOGRAPHY: The AdS/CFT correspondence suggests spacetime itself might be "
        "a quantum error-correcting code. The universe literally runs on QEC.\n\n"
        "Wheeler's 'It from Bit' becomes 'It from Qubit' -- information, specifically "
        "quantum information, may be the fundamental substrate of reality.",
    ),
    # Turn 28 -- practical wrap-up
    (
        "What resources would you recommend for someone starting to learn quantum computing?",
        "Here's a structured learning path:\n\n"
        "BOOKS:\n"
        "- 'Quantum Computing: An Applied Approach' by Jack Hidary -- best for programmers\n"
        "- 'Quantum Computation and Quantum Information' by Nielsen & Chuang -- the bible\n"
        "- 'Quantum Computing Since Democritus' by Scott Aaronson -- for the philosophy\n\n"
        "FREE COURSES:\n"
        "- IBM Qiskit Textbook (qiskit.org/textbook) -- interactive, code-first\n"
        "- MIT 8.370x on edX -- rigorous but accessible\n"
        "- Xanadu's Codebook (codebook.xanadu.ai) -- PennyLane-based, great for ML focus\n\n"
        "VIDEO:\n"
        "- 3Blue1Brown's quantum computing series -- visual intuition\n"
        "- Microsoft's Quantum Computing for Computer Scientists (YouTube) -- excellent\n\n"
        "HANDS-ON:\n"
        "- IBM Quantum Experience -- free access to real quantum hardware\n"
        "- Google's Cirq tutorials -- well-structured problem sets\n"
        "- Qiskit Global Summer School (annual, free) -- intensive 2-week program\n\n"
        "COMMUNITIES:\n"
        "- Qiskit Slack, PennyLane Forum, r/QuantumComputing\n"
        "- Quantum Open Source Foundation (QOSF) mentoring program\n\n"
        "My advice: start with the IBM Qiskit Textbook alongside Nielsen & Chuang. "
        "Write code from day one. Don't try to master all the math first -- build "
        "intuition through experimentation.",
    ),
    # Turn 29 -- quick question
    (
        "One last thing -- will quantum computing make AI smarter?",
        "The honest answer: probably not in the way people imagine. Quantum computing "
        "won't make GPT-5 smarter or train LLMs faster. Here's why:\n\n"
        "1. LLM training is mostly matrix multiplication on GPUs. Quantum computers "
        "don't offer speedups for dense matrix operations.\n"
        "2. Data loading is the bottleneck: even if you could train faster, reading "
        "training data into a quantum computer is classically expensive.\n"
        "3. Grover's speedup for search is only quadratic (sqrt(N)) -- not enough to "
        "matter for most AI tasks.\n\n"
        "WHERE QUANTUM COULD HELP AI:\n"
        "- Generating training data through quantum simulation (chemistry, materials)\n"
        "- Quantum-inspired classical algorithms (already happening: tensor networks)\n"
        "- Specific optimization subroutines within larger ML pipelines\n"
        "- Quantum generative models for certain data distributions\n\n"
        "The deeper question: could a quantum AI be fundamentally more capable? "
        "Penrose and others have speculated about quantum effects in consciousness, "
        "but this remains highly controversial and unproven. Most AI researchers "
        "don't think quantum mechanics is necessary for intelligence.",
    ),
    # Turn 30 -- final
    (
        "This was incredibly helpful. Thanks for the deep dive!",
        "You're welcome! You've covered an impressive amount of ground -- from the "
        "foundations of superposition and entanglement through error correction, "
        "algorithms, real-world applications, and even the philosophical implications. "
        "The key thing to remember: quantum computing is real, it works, and it's "
        "advancing rapidly -- but we're still in the early innings. Keep experimenting "
        "with Qiskit and PennyLane, follow the arxiv preprints, and don't believe the "
        "hype (or the doom). The future is quantum, but it's not here yet. Good luck "
        "on your quantum journey!",
    ),
]


def build_messages(num_turns: int) -> list[Message]:
    """Build a message list for the first `num_turns` turns, including the system prompt."""
    messages: list[Message] = [Message(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT)]
    for user_msg, assistant_msg in CONVERSATION_TURNS[:num_turns]:
        messages.append(Message(role=MessageRole.USER, content=user_msg))
        messages.append(Message(role=MessageRole.ASSISTANT, content=assistant_msg))
    return messages


def count_tokens(messages: list[Message]) -> int:
    """Count total tokens across all messages using Arcana's estimator."""
    total = 0
    for m in messages:
        content = m.content
        if content is None:
            continue
        if isinstance(content, str):
            total += estimate_tokens(content)
        else:
            # list[ContentBlock]
            for block in content:
                if hasattr(block, "text") and block.text:
                    total += estimate_tokens(block.text)
    return total


def main() -> None:
    console = Console()

    console.print()
    console.print(
        Panel(
            "[bold]Arcana Context Management Benchmark[/bold]\n\n"
            "Comparing naive full-context vs Arcana's WorkingSetBuilder compression.\n"
            "Simulated 30-turn quantum computing research conversation.\n"
            "No API keys required.",
            border_style="blue",
            width=72,
        )
    )
    console.print()

    # Configure the builder with a tight token budget to showcase compression.
    # In production, you'd have a larger window (128K), but the effect is the
    # same once conversation history grows large enough.
    budget = TokenBudget(
        total_window=4000,   # 4K context window (simulating a constrained budget)
        response_reserve=1500,  # Reserve 1.5K for response
        # Effective budget for input = 4000 - 1500 = 2500 tokens
    )

    builder = WorkingSetBuilder(
        identity=SYSTEM_PROMPT,
        token_budget=budget,
        goal="Help the user understand quantum computing, error correction, and practical applications",
    )

    # Build the comparison table
    table = Table(
        title="Token Usage: Naive vs Arcana",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        title_style="bold white",
        pad_edge=True,
    )
    table.add_column("Turn", justify="right", style="dim", width=5)
    table.add_column("Messages", justify="right", width=9)
    table.add_column("Naive Tokens", justify="right", width=13)
    table.add_column("Arcana Tokens", justify="right", width=14)
    table.add_column("Savings", justify="right", width=8)
    table.add_column("Compressed", justify="right", width=11)

    # Track cumulative naive cost for the summary
    sample_points = [3, 5, 8, 10, 15, 20, 25, 30]
    last_naive = 0
    last_arcana = 0
    last_compressed = 0
    last_msg_count = 0

    for turn_num in sample_points:
        messages = build_messages(turn_num)

        # Naive: count all tokens (what you'd send without compression)
        naive_tokens = count_tokens(messages)

        # Arcana: compress using WorkingSetBuilder
        compressed_messages = builder.build_conversation_context(messages, turn=turn_num)
        arcana_tokens = count_tokens(compressed_messages)

        # Get compression details from the decision
        decision = builder.last_decision
        compressed_count = decision.compressed_count if decision else 0

        # Calculate savings
        if naive_tokens > 0:
            savings_pct = (1 - arcana_tokens / naive_tokens) * 100
        else:
            savings_pct = 0.0

        # Style the savings column
        if savings_pct > 50:
            savings_style = "bold green"
        elif savings_pct > 20:
            savings_style = "yellow"
        else:
            savings_style = "dim"

        table.add_row(
            str(turn_num),
            f"{len(messages):,}",
            f"{naive_tokens:,}",
            f"{arcana_tokens:,}",
            Text(f"{savings_pct:.0f}%", style=savings_style),
            str(compressed_count),
        )

        last_naive = naive_tokens
        last_arcana = arcana_tokens
        last_compressed = compressed_count
        last_msg_count = len(messages)

    console.print(table)
    console.print()

    # Print summary
    savings_pct = (1 - last_arcana / last_naive) * 100 if last_naive > 0 else 0
    # Cost at $0.15 per million input tokens
    naive_cost = last_naive * 0.15 / 1_000_000
    arcana_cost = last_arcana * 0.15 / 1_000_000
    cost_savings = naive_cost - arcana_cost

    summary = (
        f"[bold cyan]=== Arcana Context Management ===[/bold cyan]\n\n"
        f"  Total turns:    [bold]30[/bold]\n"
        f"  Token budget:   [bold]{budget.total_window - budget.response_reserve:,}[/bold] "
        f"input tokens (of {budget.total_window:,} window)\n\n"
        f"  Naive total:    [bold]{last_naive:,}[/bold] tokens "
        f"[dim](what you'd pay without compression)[/dim]\n"
        f"  Arcana total:   [bold green]{last_arcana:,}[/bold green] tokens "
        f"[dim](with working-set compression)[/dim]\n"
        f"  Token savings:  [bold green]{savings_pct:.0f}%[/bold green]\n\n"
        f"  Messages sent:  [bold]{last_compressed}[/bold] messages compressed into summary\n"
        f"  Messages kept:  [bold]{last_msg_count - last_compressed}[/bold] verbatim "
        f"[dim](system + 6 recent + summary)[/dim]\n\n"
        f"  Naive cost:     ${naive_cost:.4f} "
        f"[dim](at $0.15/M input tokens)[/dim]\n"
        f"  Arcana cost:    ${arcana_cost:.4f}\n"
        f"  Cost savings:   [bold green]${cost_savings:.4f}[/bold green] per request\n\n"
        f"  [dim]At scale (128K window, $3/M tokens):[/dim]\n"
        f"  [dim]  1000 requests/day = ${last_naive * 3 / 1_000_000 * 1000:.2f}/day naive "
        f"vs ${last_arcana * 3 / 1_000_000 * 1000:.2f}/day with Arcana[/dim]"
    )
    console.print(Panel(summary, border_style="green", width=72))
    console.print()

    # Show the last ContextDecision for transparency
    if builder.last_decision:
        d = builder.last_decision
        detail = (
            f"[bold]Last ContextDecision (Turn 30):[/bold]\n\n"
            f"  Budget total:     {d.budget_total:,} tokens\n"
            f"  Budget used:      {d.budget_used:,} tokens "
            f"({d.budget_used * 100 // d.budget_total}% full)\n"
            f"  Messages in:      {d.messages_in}\n"
            f"  Messages out:     {d.messages_out}\n"
            f"  Compressed:       {d.compressed_count} messages\n"
            f"  History compressed: {d.history_compressed}\n"
            f"  Explanation:      {d.explanation}"
        )
        console.print(Panel(detail, title="Audit Trail", border_style="dim", width=72))
        console.print()

    # Show what compression looks like
    messages_30 = build_messages(30)
    compressed_30 = builder.build_conversation_context(messages_30, turn=30)

    console.print("[bold]What the LLM actually sees (Turn 30):[/bold]")
    console.print()
    for msg in compressed_30:
        role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        content = msg.content if isinstance(msg.content, str) else ""
        preview = content[:120].replace("\n", " ")
        if len(content) > 120:
            preview += "..."
        style = {
            "system": "bold blue",
            "user": "green",
            "assistant": "yellow",
        }.get(role_str, "white")
        role_label = Text(f"  [{role_str:>9}] ", style=style)
        preview_text = Text(preview)
        console.print(role_label + preview_text)
    console.print()


if __name__ == "__main__":
    main()

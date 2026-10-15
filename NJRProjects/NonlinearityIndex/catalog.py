"""
The Nonlinearity Index: activation function catalog data.

Each entry is built with fn(...). Fields:
  slug      URL anchor, lowercase-with-dashes
  name      display name
  family    key from FAMILIES
  f         callable x -> y, or a list of (label, callable) for several curves
  formula   HTML string, or a list of (expression, condition) pairs for piecewise functions
  desc      one or two sentences on what the curve does and why people use it
  range     output range as text
  smooth    True if differentiable everywhere (C1)
  aka       other names (optional)
  torch     torch.nn class if one exists (optional)
  src       origin, "Author et al., year" (optional)
  note      extra line shown under the plot, e.g. which parameters are drawn (optional)
  xr        x-range of the plot, default (-4, 4)
  mono      override the automatic monotonicity check (optional)
  bounded   override the automatic boundedness check (optional)

To add a function: append an fn(...) call to FUNCS and run `python3 build.py`.
"""
import math

# ---------------------------------------------------------------- helpers
def sig(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))

def softplus(x):
    return max(x, 0) + math.log1p(math.exp(-abs(x)))

def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def relu(x):
    return max(0.0, x)

def relu6(x):
    return min(max(0.0, x), 6.0)

def gelu(x):
    return x * Phi(x)

def silu(x):
    return x * sig(x)

def elu(x, a=1.0):
    return x if x > 0 else a * math.expm1(x)

def entmax15_2(x):
    if x >= 2: return 1.0
    if x <= -2: return 0.0
    u = x / 2
    return ((u + math.sqrt(2 - u * u)) / 2) ** 2

def soft_exp(a):
    def f(x):
        if a < 0:
            arg = 1 - a * (x + a)
            return None if arg <= 0 else -math.log(arg) / a
        if a == 0:
            return x
        return math.expm1(a * x) / a + a
    return f

def pau(x):
    p = 0.02 + 0.6 * x + 0.5 * x ** 2 + 0.12 * x ** 3
    q = 1 + abs(0.4 * x + 0.3 * x ** 2)
    return p / q

SELU_L, SELU_A = 1.0507009873554805, 1.6732632423543772

# ---------------------------------------------------------------- taxonomy
TIERS = [
    ("classic", "Classic", "The functions every framework ships: sigmoidal curves, rectifiers, thresholds and the output layers built on them."),
    ("modern", "Modern & learned", "Smooth self-gated units, the gated linear units inside today's Transformers, and activations whose shape is learned during training."),
    ("research", "Experimental & research", "Periodic and radial functions, the many descendants of Swish and Mish, and other proposals from the literature that are not yet mainstream."),
]

FAMILIES = [
    ("linear-step", "classic", "Linear & threshold", "The simplest maps: pass the input through, or snap it to a few discrete values."),
    ("sigmoidal", "classic", "Sigmoidal & saturating", "S-shaped curves that squash the real line into a bounded interval. Their gradients vanish once they saturate."),
    ("rectifiers", "classic", "Rectifiers (ReLU family)", "Identity for positive inputs, something cheaper or flatter for negative ones. The default hidden-layer choice since about 2011."),
    ("shrinkage", "classic", "Shrinkage & thresholding", "Functions that zero out small values, borrowed from signal denoising and sparse coding."),
    ("output", "classic", "Output & normalising", "Activations that turn logits into probabilities or positive quantities. Vector functions are plotted for the two-class case."),
    ("self-gated", "modern", "Smooth self-gated", "x multiplied by a smooth gate of itself (GELU, SiLU, Mish) and their cheaper approximations."),
    ("glu", "modern", "Gated linear units", "Two projections, one gating the other. GLU variants are plotted with both halves tied to the same input, i.e. x · g(x)."),
    ("learned", "modern", "Learned & adaptive", "Activations with trainable parameters, so every neuron or channel can learn its own shape. Plots show example parameter values."),
    ("periodic", "research", "Periodic & radial", "Oscillating and bump-shaped functions used for implicit representations, periodic signals and RBF networks."),
    ("swish-descendants", "research", "Swish & Mish descendants", "Variations on self-gating proposed since 2018, mostly evaluated on image classification benchmarks."),
    ("other-research", "research", "Other research functions", "One-off proposals and specialised activations that are worth knowing about."),
]

FUNCS = []
def fn(slug, name, family, f, formula, desc, range, smooth, **kw):
    FUNCS.append(dict(slug=slug, name=name, family=family, f=f, formula=formula, desc=desc,
                      range=range, smooth=smooth, **kw))

# ================================================================ CLASSIC
# ---------------------------------------------------------------- linear & threshold
fn("identity", "Identity", "linear-step", lambda x: x, "f(x) = x",
   "Passes the input through unchanged. It's the right output for regression and appears inside residual paths. A network built only from identity activations collapses into a single linear map.",
   "(−∞, ∞)", True, aka="Linear", torch="nn.Identity")
fn("binary-step", "Binary step", "linear-step", lambda x: 1.0 if x >= 0 else 0.0,
   [("1", "x ≥ 0"), ("0", "x < 0")],
   "Fires 1 or 0, like the original perceptron. Its derivative is zero almost everywhere, so plain gradient descent can't train it; spiking and binary networks use surrogate gradients instead.",
   "{0, 1}", False, aka="Heaviside", src="Rosenblatt, 1958")
fn("sign", "Sign", "linear-step", lambda x: (x > 0) - (x < 0), "f(x) = sign(x)",
   "Binarises activations to ±1. Binary neural networks use it in the forward pass so multiplications become bit operations, and train through it with a straight-through estimator.",
   "{−1, 0, 1}", False, aka="Signum", src="Courbariaux et al., 2016 (BinaryNet)")

# ---------------------------------------------------------------- sigmoidal
fn("sigmoid", "Sigmoid", "sigmoidal", sig, "σ(x) = 1 / (1 + e<sup>−x</sup>)",
   "Squashes any input into (0, 1), so it reads as a probability. It saturates for large |x| and its gradient peaks at 0.25, which made deep sigmoid networks hard to train. Still the standard choice for binary outputs and LSTM/GRU gates.",
   "(0, 1)", True, aka="Logistic", torch="nn.Sigmoid")
fn("tanh", "Tanh", "sigmoidal", math.tanh, "tanh(x) = (e<sup>x</sup> − e<sup>−x</sup>) / (e<sup>x</sup> + e<sup>−x</sup>)",
   "A rescaled sigmoid (tanh x = 2σ(2x) − 1) that is zero-centred, which helps optimisation. It saturates like the sigmoid. Standard for recurrent hidden states and for squashing bounded outputs.",
   "(−1, 1)", True, aka="Hyperbolic tangent", torch="nn.Tanh")
fn("lecun-tanh", "LeCun tanh", "sigmoidal", lambda x: 1.7159 * math.tanh(2 * x / 3), "f(x) = 1.7159 · tanh(2x / 3)",
   "Tanh rescaled so that f(±1) = ±1, which keeps unit-variance inputs at roughly unit variance and keeps the units away from saturation.",
   "(−1.7159, 1.7159)", True, aka="Scaled tanh", src="LeCun et al., 1998")
fn("arctan", "Arctan", "sigmoidal", math.atan, "f(x) = arctan(x)",
   "Shaped like tanh but approaches its limits polynomially rather than exponentially, so it saturates more gently and keeps a usable gradient further from zero.",
   "(−π/2, π/2)", True, aka="Inverse tangent")
fn("softsign", "Softsign", "sigmoidal", lambda x: x / (1 + abs(x)), "f(x) = x / (1 + |x|)",
   "A tanh-like curve with polynomial tails: it saturates more slowly than tanh, so gradients survive further from the origin. Studied as a tanh replacement in deep feed-forward nets.",
   "(−1, 1)", True, aka="Elliott function", torch="nn.Softsign", src="Elliott, 1993; Glorot & Bengio, 2010")
fn("hard-sigmoid", "Hard sigmoid", "sigmoidal", lambda x: max(0.0, min(1.0, x / 6 + 0.5)),
   [("0", "x ≤ −3"), ("x / 6 + 1/2", "−3 < x < 3"), ("1", "x ≥ 3")],
   "A piecewise-linear stand-in for the sigmoid. It's cheap and quantisation-friendly, which is why MobileNetV3 uses it in squeeze-and-excitation blocks. Some libraries use a slope of 0.2 instead of 1/6.",
   "[0, 1]", False, torch="nn.Hardsigmoid", xr=(-5, 5))
fn("hard-tanh", "Hard tanh", "sigmoidal", lambda x: max(-1.0, min(1.0, x)), "f(x) = max(−1, min(1, x))",
   "Clips to [−1, 1] with slope 1 in between. The gradient is exactly zero outside the linear zone; binary networks use its derivative as the straight-through estimator.",
   "[−1, 1]", False, torch="nn.Hardtanh")
fn("log-sigmoid", "Log-sigmoid", "sigmoidal", lambda x: -softplus(-x), "f(x) = log σ(x) = −softplus(−x)",
   "The logarithm of the sigmoid, computed stably. It mostly appears inside losses (binary cross-entropy, word2vec negative sampling) rather than as a hidden-layer activation.",
   "(−∞, 0)", True, torch="nn.LogSigmoid")
fn("erf", "Error function", "sigmoidal", math.erf, "erf(x) = (2/√π) ∫<sub>0</sub><sup>x</sup> e<sup>−t²</sup> dt",
   "The Gauss error function: the normal CDF rescaled to (−1, 1). It behaves like tanh but saturates faster, because its tails are Gaussian.",
   "(−1, 1)", True, aka="erf")
fn("probit", "Probit (normal CDF)", "sigmoidal", Phi, "Φ(x) = ½ [1 + erf(x / √2)]",
   "The cumulative distribution of a standard normal. As an output link it gives probit regression; inside a network it's the gate in GELU.",
   "(0, 1)", True, aka="Φ, Gaussian CDF")
fn("cloglog", "Complementary log-log", "sigmoidal", lambda x: -math.expm1(-math.exp(x)), "f(x) = 1 − exp(−e<sup>x</sup>)",
   "An asymmetric sigmoid: it leaves 0 slowly and approaches 1 quickly. It's the natural link for discrete-time survival models with proportional hazards, and for rare-event binary outcomes.",
   "(0, 1)", True, aka="cloglog, Gompertz link", bounded=True)
fn("gudermannian", "Gudermannian", "sigmoidal", lambda x: 2 * math.atan(math.tanh(x / 2)), "gd(x) = 2 arctan(tanh(x / 2))",
   "Connects circular and hyperbolic functions. Another smooth, odd, saturating curve between tanh and arctan in how fast it saturates.",
   "(−π/2, π/2)", True, aka="gd")
fn("isru", "ISRU", "sigmoidal", lambda x: x / math.sqrt(1 + x * x), "f(x) = x / √(1 + αx²)",
   "Inverse square root unit: tanh-shaped, but needs only a square root instead of exponentials, which is cheaper on some hardware. α sets the bounds ±1/√α (α = 1 plotted).",
   "(−1/√α, 1/√α)", True, aka="Inverse square root unit", src="Carlile et al., 2017")

# ---------------------------------------------------------------- rectifiers
fn("relu", "ReLU", "rectifiers", relu, "f(x) = max(0, x)",
   "Zero for negative inputs, identity for positive ones. It's cheap, doesn't saturate for x > 0 and produces sparse activations, which made deep networks practical. A unit that only ever sees negative inputs stops learning: a 'dead' ReLU.",
   "[0, ∞)", False, aka="Rectified linear unit", torch="nn.ReLU", src="Nair & Hinton, 2010; Glorot et al., 2011")
fn("leaky-relu", "Leaky ReLU", "rectifiers", lambda x: x if x >= 0 else 0.01 * x,
   [("x", "x ≥ 0"), ("αx", "x < 0")],
   "Gives negative inputs a small slope α (0.01 by default) so units never fully die. GAN discriminators and graph attention networks typically use α = 0.2.",
   "(−∞, ∞)", False, aka="LReLU", torch="nn.LeakyReLU", src="Maas et al., 2013")
fn("rrelu", "RReLU", "rectifiers", lambda x: x if x >= 0 else (1 / 8 + 1 / 3) / 2 * x,
   [("x", "x ≥ 0"), ("ax,  a ~ U(1/8, 1/3) in training", "x < 0")],
   "A leaky ReLU whose negative slope is sampled at random during training, which adds a little regularising noise. At test time the slope is fixed to the mean, (l + u)/2 ≈ 0.23, which is what's plotted.",
   "(−∞, ∞)", False, aka="Randomised leaky ReLU", torch="nn.RReLU", src="Xu et al., 2015")
fn("relu6", "ReLU6", "rectifiers", relu6, "f(x) = min(max(0, x), 6)",
   "ReLU capped at 6. The bounded range keeps activations representable in low-precision fixed-point arithmetic, which is why mobile networks such as MobileNetV2 adopted it.",
   "[0, 6]", False, torch="nn.ReLU6", src="Krizhevsky, 2010; Howard et al., 2017", xr=(-3, 8))
fn("thresholded-relu", "Thresholded ReLU", "rectifiers", lambda x: x if x > 1 else 0.0,
   [("x", "x > θ"), ("0", "otherwise")],
   "Passes values only above a threshold θ (θ = 1 plotted). It was introduced for zero-bias autoencoders, where it gives sparse codes without learned biases.",
   "{0} ∪ (θ, ∞)", False, torch="nn.Threshold", src="Konda et al., 2015")
fn("absolute", "Absolute value", "rectifiers", abs, "f(x) = |x|",
   "Rectifies both signs, so it discards the sign but keeps the magnitude. Early multi-stage vision systems used absolute-value rectification of filter responses.",
   "[0, ∞)", False, aka="Abs rectification", src="Jarrett et al., 2009")
fn("crelu", "CReLU", "rectifiers", [("ReLU(x)", relu), ("ReLU(−x)", lambda x: relu(-x))],
   "f(x) = [ReLU(x), ReLU(−x)]",
   "Concatenates ReLU(x) and ReLU(−x), doubling the channels so no information is thrown away. It was motivated by the observation that early CNN filters tend to learn negated pairs.",
   "[0, ∞)²", False, aka="Concatenated ReLU", src="Shang et al., 2016", mono=False)
fn("softplus", "Softplus", "rectifiers", softplus, "f(x) = ln(1 + e<sup>x</sup>)",
   "A smooth ReLU: always positive and differentiable everywhere. Common as an output for quantities that must be positive (variances, rates), and the inner function of Mish.",
   "(0, ∞)", True, torch="nn.Softplus", src="Dugas et al., 2001")
fn("elu", "ELU", "rectifiers", elu, [("x", "x > 0"), ("α(e<sup>x</sup> − 1)", "x ≤ 0")],
   "Identity for positive inputs and a smooth exponential that saturates at −α for negative ones (α = 1 plotted). The negative outputs push mean activations toward zero, which speeds up learning.",
   "(−α, ∞)", True, aka="Exponential linear unit", torch="nn.ELU", src="Clevert et al., 2015")
fn("selu", "SELU", "rectifiers", lambda x: SELU_L * (x if x > 0 else SELU_A * math.expm1(x)),
   [("λx", "x > 0"), ("λα(e<sup>x</sup> − 1)", "x ≤ 0")],
   "A scaled ELU (λ ≈ 1.0507, α ≈ 1.6733) whose constants make activations converge to zero mean and unit variance across layers, given LeCun-normal initialisation: 'self-normalising' networks that don't need batch norm.",
   "(−λα, ∞) ≈ (−1.76, ∞)", False, aka="Scaled ELU", torch="nn.SELU", src="Klambauer et al., 2017")
fn("celu", "CELU", "rectifiers",
   [("α = 0.5", lambda x: max(0, x) + min(0, 0.5 * math.expm1(x / 0.5))),
    ("α = 1", lambda x: max(0, x) + min(0, math.expm1(x))),
    ("α = 2", lambda x: max(0, x) + min(0, 2 * math.expm1(x / 2)))],
   "f(x) = max(0, x) + min(0, α(e<sup>x/α</sup> − 1))",
   "A version of ELU that is continuously differentiable for every α. Plain ELU's derivative jumps at 0 whenever α ≠ 1. Plotted for α = 0.5, 1 and 2.",
   "(−α, ∞)", True, aka="Continuously differentiable ELU", torch="nn.CELU", src="Barron, 2017", mono=True)
fn("isrlu", "ISRLU", "rectifiers", lambda x: x if x >= 0 else x / math.sqrt(1 + x * x),
   [("x", "x ≥ 0"), ("x / √(1 + αx²)", "x < 0")],
   "ELU-shaped, but the negative side uses an inverse square root instead of an exponential, which is cheaper to compute (α = 1 plotted).",
   "(−1/√α, ∞)", True, aka="Inverse square root linear unit", src="Carlile et al., 2017")
fn("shifted-softplus", "Shifted softplus", "rectifiers", lambda x: softplus(x) - math.log(2), "f(x) = ln(0.5e<sup>x</sup> + 0.5)",
   "Softplus shifted down by ln 2 so that f(0) = 0. SchNet uses it so that predicted molecular energies, and the forces derived from them, are smooth.",
   "(−ln 2, ∞)", True, aka="ssp", src="Schütt et al., 2017 (SchNet)")

# ---------------------------------------------------------------- shrinkage
fn("hardshrink", "Hard shrink", "shrinkage", lambda x: x if abs(x) > 0.5 else 0.0,
   [("x", "|x| > λ"), ("0", "otherwise")],
   "Zeroes inputs inside [−λ, λ] and passes the rest unchanged (λ = 0.5 plotted). This is hard thresholding, as used in wavelet denoising and sparse coding.",
   "{0} ∪ (−∞, −λ) ∪ (λ, ∞)", False, torch="nn.Hardshrink", xr=(-3, 3))
fn("softshrink", "Soft shrink", "shrinkage", lambda x: x - 0.5 if x > 0.5 else (x + 0.5 if x < -0.5 else 0.0),
   [("x − λ", "x > λ"), ("x + λ", "x < −λ"), ("0", "otherwise")],
   "Soft thresholding: zero inside [−λ, λ], and larger values move toward zero by λ. It's the proximal operator of the L1 norm, which is why it appears in unrolled sparse-coding networks such as LISTA.",
   "(−∞, ∞)", False, torch="nn.Softshrink", xr=(-3, 3))
fn("tanhshrink", "Tanh shrink", "shrinkage", lambda x: x - math.tanh(x), "f(x) = x − tanh(x)",
   "What's left after subtracting tanh: almost flat near zero and nearly linear for large |x|. Rarely used in practice, but available in PyTorch.",
   "(−∞, ∞)", True, torch="nn.Tanhshrink")

# ---------------------------------------------------------------- output & normalising
fn("softmax", "Softmax", "output",
   [("T = 0.5", lambda x: sig(x / 0.5)), ("T = 1", sig), ("T = 2", lambda x: sig(x / 2))],
   "softmax(z)<sub>i</sub> = e<sup>z<sub>i</sub>/T</sup> / Σ<sub>j</sub> e<sup>z<sub>j</sub>/T</sup>",
   "Turns a vector of logits into a probability distribution. With two classes it reduces to the sigmoid of the logit difference, which is what's plotted against x = z₁ − z₂ for temperatures T = 0.5, 1 and 2. Lower T gives a sharper distribution.",
   "(0, 1), sums to 1", True, aka="Normalised exponential", torch="nn.Softmax", mono=True, note="p₁ for two classes, x = z₁ − z₂")
fn("log-softmax", "Log-softmax", "output", lambda x: -softplus(-x),
   "f(z)<sub>i</sub> = z<sub>i</sub> − log Σ<sub>j</sub> e<sup>z<sub>j</sub></sup>",
   "Log of softmax, computed in one numerically stable step with log-sum-exp. It's paired with negative log-likelihood loss; most cross-entropy implementations use it internally.",
   "(−∞, 0)", True, torch="nn.LogSoftmax", note="log p₁ for two classes, x = z₁ − z₂")
fn("softmin", "Softmin", "output", lambda x: sig(-x), "softmin(z) = softmax(−z)",
   "Softmax of the negated inputs, so the smallest value gets the most weight. Used for soft nearest-neighbour assignments and differentiable minimums.",
   "(0, 1), sums to 1", True, torch="nn.Softmin", note="p₁ for two classes, x = z₁ − z₂")
fn("sparsemax", "Sparsemax", "output", lambda x: min(1.0, max(0.0, (x + 1) / 2)),
   "sparsemax(z) = argmin<sub>p ∈ Δ</sub> ‖p − z‖²",
   "The Euclidean projection of the logits onto the probability simplex. Unlike softmax it can output exact zeros, which gives sparse, more interpretable attention. For two classes it's a hard sigmoid.",
   "[0, 1], sums to 1", False, src="Martins & Astudillo, 2016", note="p₁ for two classes, x = z₁ − z₂")
fn("entmax15", "1.5-entmax", "output", entmax15_2,
   "argmax<sub>p ∈ Δ</sub> pᵀz + H<sub>α</sub>(p),  α = 1.5",
   "Sits between softmax (α = 1) and sparsemax (α = 2): smooth near ties, but it can still assign exact zeros. Used for sparse attention in sequence-to-sequence models.",
   "[0, 1], sums to 1", True, aka="Entmax", src="Peters et al., 2019", note="p₁ for two classes, x = z₁ − z₂")
fn("exponential", "Exponential", "output", math.exp, "f(x) = e<sup>x</sup>",
   "Maps to strictly positive values. Used as an output for rates and scale parameters (a Poisson intensity, a standard deviation); its unbounded growth makes it unstable in hidden layers.",
   "(0, ∞)", True, xr=(-4, 2.5))

# ================================================================ MODERN & LEARNED
# ---------------------------------------------------------------- smooth self-gated
fn("gelu", "GELU", "self-gated", gelu, "f(x) = x · Φ(x)",
   "Weights each input by the probability that a standard normal variable falls below it. It's smooth and dips slightly below zero (minimum ≈ −0.17 near x = −0.75). The default in BERT, GPT-2/3 and most Transformers before GLU variants took over.",
   "≈ [−0.17, ∞)", True, aka="Gaussian error linear unit", torch="nn.GELU", src="Hendrycks & Gimpel, 2016")
fn("gelu-tanh", "GELU (tanh approximation)", "self-gated",
   lambda x: 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3))),
   "f(x) = ½x [1 + tanh(√(2/π) (x + 0.044715x³))]",
   "The fast approximation used in the original BERT and GPT-2 code. It's visually indistinguishable from exact GELU; pick whichever the pretrained weights you load were trained with.",
   "≈ [−0.17, ∞)", True, torch="nn.GELU(approximate='tanh')", src="Hendrycks & Gimpel, 2016")
fn("quick-gelu", "QuickGELU", "self-gated", lambda x: x * sig(1.702 * x), "f(x) = x · σ(1.702x)",
   "A cheaper sigmoid-based approximation of GELU. OpenAI's CLIP models were trained with it, so loading their weights means using it too.",
   "≈ [−0.16, ∞)", True, aka="Sigmoid GELU approximation", src="Hendrycks & Gimpel, 2016; Radford et al., 2021 (CLIP)")
fn("silu", "SiLU / Swish", "self-gated", silu, "f(x) = x · σ(x)",
   "The input multiplied by its own sigmoid. Smooth and non-monotonic, with a minimum of about −0.28 at x ≈ −1.28. Used in EfficientNet, the YOLO family and most diffusion-model U-Nets.",
   "≈ [−0.28, ∞)", True, aka="Swish-1, sigmoid-weighted linear unit", torch="nn.SiLU",
   src="Elfwing et al., 2017; Ramachandran et al., 2017")
fn("hard-swish", "Hard swish", "self-gated", lambda x: x * relu6(x + 3) / 6,
   [("0", "x ≤ −3"), ("x(x + 3) / 6", "−3 < x < 3"), ("x", "x ≥ 3")],
   "A piecewise-polynomial approximation of Swish with no exponentials, so it runs fast on mobile CPUs and quantises well. Introduced in MobileNetV3.",
   "≈ [−0.375, ∞)", False, aka="h-swish", torch="nn.Hardswish", src="Howard et al., 2019", xr=(-5, 4))
fn("mish", "Mish", "self-gated", lambda x: x * math.tanh(softplus(x)), "f(x) = x · tanh(softplus(x))",
   "Smooth and non-monotonic, unbounded above and bounded below (minimum ≈ −0.31). Very similar to Swish with a slightly deeper negative lobe. Used in the YOLOv4 backbone.",
   "≈ [−0.31, ∞)", True, torch="nn.Mish", src="Misra, 2019")
fn("squareplus", "Squareplus", "self-gated", lambda x: (x + math.sqrt(x * x + 4)) / 2, "f(x) = (x + √(x² + b)) / 2",
   "An algebraic softplus: no exponentials or logarithms, so it's fast and numerically simple. b controls the curvature near zero (b = 4 plotted).",
   "(0, ∞)", True, src="Barron, 2021")
fn("smelu", "SmeLU", "self-gated", lambda x: 0.0 if x <= -1 else ((x + 1) ** 2 / 4 if x < 1 else x),
   [("0", "x ≤ −β"), ("(x + β)² / 4β", "|x| < β"), ("x", "x ≥ β")],
   "Smooth ReLU: replaces ReLU's corner with a quadratic patch of width 2β (β = 1 plotted). Proposed to make recommender-system predictions more reproducible between training runs.",
   "[0, ∞)", True, aka="Smooth ReLU", src="Shamir et al., 2020")
fn("squared-relu", "Squared ReLU", "self-gated", lambda x: relu(x) ** 2, "f(x) = max(0, x)²",
   "ReLU squared. Primer's architecture search found it makes Transformer language models train more efficiently, and later efficient-LLM work adopted it.",
   "[0, ∞)", True, aka="ReLU²", src="So et al., 2021 (Primer)", xr=(-3, 3))
fn("star-relu", "StarReLU", "self-gated", lambda x: 0.8944 * relu(x) ** 2 - 0.4472, "f(x) = s · ReLU(x)² + b",
   "Squared ReLU with a learnable scale and bias (initialised to s ≈ 0.894, b ≈ −0.447) that keep the output distribution normalised. Cheaper than GELU with similar accuracy in MetaFormer models.",
   "[b, ∞)", True, src="Yu et al., 2022 (MetaFormer baselines)", xr=(-3, 3))

# ---------------------------------------------------------------- GLU family
fn("glu", "GLU", "glu", lambda x: x * sig(x), "GLU(a, b) = a ⊗ σ(b)",
   "Splits a projection into a value half a and a gate half b and multiplies a by σ(b), so the network controls how much information flows. Introduced for convolutional language models.",
   "(−∞, ∞)", True, aka="Gated linear unit", torch="nn.GLU", src="Dauphin et al., 2017", mono=False)
fn("bilinear", "Bilinear", "glu", lambda x: x * x, "f(a, b) = a ⊗ b",
   "The gate without any nonlinearity: just the elementwise product of two projections. It was surprisingly competitive in Transformer feed-forward experiments, which suggests the multiplication matters more than the exact gate.",
   "(−∞, ∞)", True, src="Shazeer, 2020", mono=False, xr=(-3, 3))
fn("reglu", "ReGLU", "glu", lambda x: x * relu(x), "ReGLU(a, b) = a ⊗ ReLU(b)",
   "GLU with a ReLU gate. It beat standard ReLU feed-forward layers at equal compute, and FT-Transformer uses it for tabular data.",
   "(−∞, ∞)", False, src="Shazeer, 2020", mono=False, xr=(-3, 3))
fn("geglu", "GEGLU", "glu", lambda x: x * gelu(x), "GEGLU(a, b) = a ⊗ GELU(b)",
   "GLU with a GELU gate. Used in T5 v1.1, Gemma and other Transformer language models.",
   "(−∞, ∞)", True, src="Shazeer, 2020", mono=False, xr=(-3, 3))
fn("swiglu", "SwiGLU", "glu", lambda x: x * silu(x), "SwiGLU(a, b) = a ⊗ Swish(b)",
   "The feed-forward activation of PaLM, LLaMA and most open LLMs. Because it needs three weight matrices instead of two, the hidden size is usually cut to about 2/3 to keep the parameter count equal.",
   "(−∞, ∞)", True, src="Shazeer, 2020", mono=False, xr=(-3, 3))
fn("gated-tanh", "Gated tanh unit", "glu", lambda x: math.tanh(x) * sig(x), "f(a, b) = tanh(a) ⊗ σ(b)",
   "A bounded tanh 'content' path modulated by a sigmoid gate. The activation inside WaveNet and gated PixelCNN for audio and image generation.",
   "(−1, 1)", True, aka="WaveNet gate, GTU", src="van den Oord et al., 2016", mono=False)

# ---------------------------------------------------------------- learned & adaptive
fn("prelu", "PReLU", "learned",
   [("α = 0", relu), ("α = 0.25", lambda x: x if x >= 0 else 0.25 * x), ("α = 0.5", lambda x: x if x >= 0 else 0.5 * x)],
   [("x", "x ≥ 0"), ("αx,  α learned", "x < 0")],
   "A leaky ReLU whose negative slope α is learned, usually one per channel and initialised to 0.25. It was used in the first network reported to beat human-level accuracy on ImageNet.",
   "(−∞, ∞)", False, aka="Parametric ReLU", torch="nn.PReLU", src="He et al., 2015", mono=True)
fn("swish-beta", "Swish-β", "learned",
   [("β = 0.5", lambda x: x * sig(0.5 * x)), ("β = 1", silu), ("β = 4", lambda x: x * sig(4 * x))],
   "f(x) = x · σ(βx),  β learned",
   "Swish with a trainable β that interpolates between a scaled linear function (β → 0 gives x/2) and ReLU (β → ∞). Plotted for β = 0.5, 1 and 4.",
   "(−∞, ∞)", True, aka="Parametric Swish", src="Ramachandran et al., 2017", mono=False)
fn("pelu", "PELU", "learned",
   [("b = 0.5", lambda x: x / 0.5 if x >= 0 else math.expm1(x / 0.5)),
    ("b = 1", lambda x: x if x >= 0 else math.expm1(x)),
    ("b = 2", lambda x: x / 2 if x >= 0 else math.expm1(x / 2))],
   [("(a / b) x", "x ≥ 0"), ("a(e<sup>x/b</sup> − 1)", "x < 0")],
   "ELU with two learned positive parameters: a sets the saturation level and b the scale. Tying them as a/b keeps the function differentiable at 0. Plotted with a = 1 and b = 0.5, 1, 2.",
   "(−a, ∞)", True, aka="Parametric ELU", src="Trottier et al., 2017", mono=True)
fn("srelu", "SReLU", "learned",
   lambda x: (1.5 + 0.4 * (x - 1.5)) if x >= 1.5 else ((-1 + 0.2 * (x + 1)) if x <= -1 else x),
   [("t<sub>r</sub> + a<sub>r</sub>(x − t<sub>r</sub>)", "x ≥ t<sub>r</sub>"), ("x", "t<sub>l</sub> < x < t<sub>r</sub>"),
    ("t<sub>l</sub> + a<sub>l</sub>(x − t<sub>l</sub>)", "x ≤ t<sub>l</sub>")],
   "Three linear pieces with four learnable parameters (two thresholds, two outer slopes), so it can learn both convex and non-convex shapes. Inspired by psychophysical laws such as Weber–Fechner.",
   "(−∞, ∞)", False, aka="S-shaped ReLU", src="Jin et al., 2016",
   note="t_l = −1, a_l = 0.2, t_r = 1.5, a_r = 0.4")
fn("apl", "APL", "learned", lambda x: relu(x) + 0.5 * relu(1 - x),
   "f(x) = max(0, x) + Σ<sub>s</sub> a<sub>s</sub> max(0, −x + b<sub>s</sub>)",
   "Adaptive piecewise linear unit: a ReLU plus a sum of learned hinge functions, so each neuron learns its own piecewise-linear shape. Plotted with one hinge (a = 0.5, b = 1).",
   "learned", False, aka="Adaptive piecewise linear", src="Agostinelli et al., 2014")
fn("maxout", "Maxout", "learned", lambda x: max(-0.6 * x - 1, 0.25 * x, 1.4 * x - 1.2),
   "f(x) = max<sub>j</sub> (w<sub>j</sub>ᵀx + b<sub>j</sub>)",
   "Outputs the maximum of k learned linear functions, so it can approximate any convex function; ReLU and leaky ReLU are special cases. It costs k times the parameters. Plotted with three example pieces.",
   "learned", False, src="Goodfellow et al., 2013", mono=False)
fn("acon-c", "ACON-C", "learned", lambda x: 0.75 * x * sig(0.75 * x) + 0.25 * x,
   "f(x) = (p₁ − p₂) x · σ(β(p₁ − p₂) x) + p₂x",
   "'Activate or not': a smooth maximum of p₁x and p₂x with learnable p₁, p₂ and switching factor β. With p₁ = 1 and p₂ = 0 it's exactly Swish; Meta-ACON predicts β from the input.",
   "learned", True, src="Ma et al., 2021", note="p₁ = 1, p₂ = 0.25, β = 1")
fn("pau", "Padé activation unit", "learned", pau,
   "f(x) = P(x) / (1 + |Σ<sub>k</sub> b<sub>k</sub>x<sup>k</sup>|)",
   "A learnable rational function (a ratio of polynomials) whose denominator can't reach zero. It can closely approximate standard activations or learn new shapes. Plotted with example coefficients.",
   "learned", True, aka="PAU, rational activation", src="Molina et al., 2020", note="example coefficients")
fn("soft-exponential", "Soft exponential", "learned",
   [("α = −0.5", soft_exp(-0.5)), ("α = 0", soft_exp(0)), ("α = 0.5", soft_exp(0.5))],
   [("−ln(1 − α(x + α)) / α", "α < 0"), ("x", "α = 0"), ("(e<sup>αx</sup> − 1) / α + α", "α > 0")],
   "One parameter α moves continuously between logarithmic (α < 0), linear (α = 0) and exponential (α > 0) behaviour, so a neuron can learn to compute sums, products or exponentials.",
   "depends on α", True, src="Godfrey & Gashler, 2015", mono=True)

# ================================================================ EXPERIMENTAL & RESEARCH
# ---------------------------------------------------------------- periodic & radial
fn("sine", "Sine (SIREN)", "periodic", math.sin, "f(x) = sin(ω₀x)",
   "A periodic activation for coordinate-based networks. With ω₀ = 30 and a matching initialisation, SIRENs represent images, audio and signed distance fields in fine detail, and their derivatives are SIRENs too. Plotted with ω₀ = 1.",
   "[−1, 1]", True, aka="Sinusoid", src="Sitzmann et al., 2020 (SIREN)", xr=(-6.3, 6.3))
fn("cosine", "Cosine", "periodic", math.cos, "f(x) = cos(x)",
   "Periodic and even. Appears in random Fourier-feature networks and as an alternative to sine in implicit neural representations.",
   "[−1, 1]", True, xr=(-6.3, 6.3))
fn("sinc", "Sinc", "periodic", lambda x: 1.0 if x == 0 else math.sin(x) / x, "f(x) = sin(x) / x",
   "A decaying oscillation whose Fourier transform is a box: an ideal low-pass filter in function form. Studied as an activation for function approximation.",
   "≈ [−0.217, 1]", True, xr=(-10, 10))
fn("snake", "Snake", "periodic", lambda x: x + math.sin(x) ** 2, "f(x) = x + sin²(ax) / a",
   "The identity plus a periodic ripple, which gives networks an inductive bias toward periodic signals while staying monotonic. BigVGAN uses a version with learnable a for audio synthesis. Plotted with a = 1.",
   "(−∞, ∞)", True, src="Ziyin et al., 2020")
fn("gcu", "Growing cosine unit", "periodic", lambda x: x * math.cos(x), "f(x) = x · cos(x)",
   "Oscillates with growing amplitude. Because it is non-monotonic with many zeros, a single GCU neuron can learn XOR.",
   "(−∞, ∞)", True, aka="GCU", src="Noel et al., 2021", xr=(-6.3, 6.3))
fn("sinlu", "SinLU", "periodic", lambda x: (x + math.sin(x)) * sig(x), "f(x) = (x + a sin(bx)) · σ(x)",
   "A Swish-like gated unit that adds a learned sinusoidal ripple to the input before gating. Plotted with a = b = 1.",
   "≈ [−0.3, ∞)", True, aka="Sinu-sigmoidal linear unit", src="Paul et al., 2022")
fn("gaussian", "Gaussian", "periodic", lambda x: math.exp(-x * x), "f(x) = e<sup>−x²</sup>",
   "A bump that responds only near zero. The classic activation of radial basis function networks, and also used in implicit neural representations.",
   "(0, 1]", True, aka="RBF", xr=(-3, 3))
fn("ricker", "Ricker wavelet", "periodic", lambda x: (1 - x * x) * math.exp(-x * x / 2), "f(x) = (1 − x²) e<sup>−x²/2</sup>",
   "The 'Mexican hat': a negated second derivative of a Gaussian. Wavelet neural networks use it to capture localised, multi-scale features.",
   "[−2e<sup>−3/2</sup>, 1] ≈ [−0.45, 1]", True, aka="Mexican hat")
fn("multiquadric", "Multiquadric", "periodic", lambda x: math.sqrt(x * x + 1), "f(x) = √(x² + c²)",
   "A radial basis function that grows with distance from the centre (c = 1 plotted). Used in RBF interpolation and some RBF networks.",
   "[c, ∞)", True)
fn("inverse-multiquadric", "Inverse multiquadric", "periodic", lambda x: 1 / math.sqrt(x * x + 1), "f(x) = 1 / √(x² + c²)",
   "A heavy-tailed bump: it decays like 1/|x| rather than exponentially, so each unit responds to a wider neighbourhood than a Gaussian (c = 1 plotted).",
   "(0, 1/c]", True)

# ---------------------------------------------------------------- Swish & Mish descendants
fn("e-swish", "E-swish", "swish-descendants", lambda x: 1.375 * x * sig(x), "f(x) = βx · σ(x)",
   "Swish multiplied by a constant β between 1 and 2 (β = 1.375 plotted), which its author found trained faster on image classification.",
   "≈ [−0.38, ∞)", True, src="Alcaide, 2018")
fn("elish", "ELiSH", "swish-descendants", lambda x: x * sig(x) if x >= 0 else math.expm1(x) * sig(x),
   [("x · σ(x)", "x ≥ 0"), ("(e<sup>x</sup> − 1) · σ(x)", "x < 0")],
   "Swish for positive inputs and an ELU-style exponential times the sigmoid for negative ones. Found with a genetic search over activation shapes.",
   "≈ [−0.24, ∞)", True, aka="Exponential linear sigmoid squashing", src="Basirat & Roth, 2018")
fn("hard-elish", "HardELiSH", "swish-descendants",
   lambda x: (x if x >= 0 else math.expm1(x)) * max(0.0, min(1.0, (x + 1) / 2)),
   [("x · max(0, min(1, (x + 1)/2))", "x ≥ 0"), ("(e<sup>x</sup> − 1) · max(0, min(1, (x + 1)/2))", "x < 0")],
   "The hard-sigmoid version of ELiSH, cheaper to compute.",
   "≈ [−0.12, ∞)", False, src="Basirat & Roth, 2018", mono=False)
fn("tanhexp", "TanhExp", "swish-descendants", lambda x: x * math.tanh(math.exp(x)), "f(x) = x · tanh(e<sup>x</sup>)",
   "Close to linear for positive inputs, because tanh(eˣ) reaches 1 quickly, with a small negative lobe. Proposed for lightweight networks.",
   "≈ [−0.35, ∞)", True, src="Liu & Di, 2020")
fn("serf", "Serf", "swish-descendants", lambda x: x * math.erf(softplus(x)), "f(x) = x · erf(ln(1 + e<sup>x</sup>))",
   "Log-softplus error function: Mish's shape with erf in place of tanh. Reported gains over Mish and Swish on image benchmarks.",
   "≈ [−0.31, ∞)", True, src="Nag & Bhattacharyya, 2021")
fn("logish", "Logish", "swish-descendants", lambda x: x * math.log1p(sig(x)), "f(x) = x · ln(1 + σ(x))",
   "Gates the input with the logarithm of 1 + σ(x); a gentler gate than Swish's, saturating at ln 2.",
   "≈ [−0.11, ∞)", True, src="Zhu et al., 2021")
fn("smish", "Smish", "swish-descendants", lambda x: x * math.tanh(math.log1p(sig(x))), "f(x) = x · tanh(ln(1 + σ(x)))",
   "Combines ideas from Mish and Logish: tanh of the log-sigmoid gate.",
   "≈ [−0.11, ∞)", True, src="Wang et al., 2022")
fn("lisht", "LiSHT", "swish-descendants", lambda x: x * math.tanh(x), "f(x) = x · tanh(x)",
   "Even and non-negative: negative inputs map to positive outputs. Its authors argue the large gradients on the negative side avoid dying neurons.",
   "[0, ∞)", True, aka="Linearly scaled hyperbolic tangent", src="Roy et al., 2019", xr=(-3, 3))

# ---------------------------------------------------------------- other research
fn("bent-identity", "Bent identity", "other-research", lambda x: (math.sqrt(x * x + 1) - 1) / 2 + x,
   "f(x) = (√(x² + 1) − 1) / 2 + x",
   "Almost linear with a gentle bend: the slope goes from 0.5 for large negative inputs to 1.5 for large positive ones. Smooth, monotonic and unbounded.",
   "(−∞, ∞)", True)
fn("sqnl", "SQNL", "other-research",
   lambda x: 1.0 if x > 2 else (x - x * x / 4 if x >= 0 else (x + x * x / 4 if x >= -2 else -1.0)),
   [("1", "x > 2"), ("x − x²/4", "0 ≤ x ≤ 2"), ("x + x²/4", "−2 ≤ x < 0"), ("−1", "x < −2")],
   "Square nonlinearity: a tanh-shaped curve built from two quadratic pieces, cheap enough for hardware without exponential units.",
   "[−1, 1]", True, src="Wuraola & Patel, 2018")
fn("penalized-tanh", "Penalised tanh", "other-research", lambda x: math.tanh(x) if x > 0 else 0.25 * math.tanh(x),
   [("tanh(x)", "x > 0"), ("a · tanh(x)", "x ≤ 0")],
   "Tanh with the negative side scaled down (a = 0.25), the saturating analogue of a leaky ReLU. A broad NLP comparison later found it unusually stable across tasks.",
   "(−a, 1)", False, src="Xu et al., 2016; Eger et al., 2018")
fn("cube", "Cube", "other-research", lambda x: x ** 3, "f(x) = x³",
   "Used in a neural dependency parser to model three-way interactions between word, part-of-speech and label features directly. Unbounded and fast-growing, so it needs careful scaling.",
   "(−∞, ∞)", True, src="Chen & Manning, 2014", xr=(-2, 2))

# Well-known functions queued for the next batches (shown on the site as "coming next").
QUEUE = ["Dynamic ReLU", "FReLU (funnel activation)", "Kernel activation functions (KAF)", "Hexpo",
         "Soft clipping", "ErfAct", "SMU", "Meta-ACON", "Bipolar ReLU", "Adaptive blending units",
         "KAN spline activations", "Rectified power units (RePU)"]

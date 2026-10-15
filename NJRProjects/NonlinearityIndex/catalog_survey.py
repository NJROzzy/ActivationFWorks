"""
Activation functions added from the survey
  V. Kunc & J. Kléma, "Three Decades of Activations: A Comprehensive Survey of 400 Activation
  Functions for Neural Networks", arXiv:2402.09092, 2024.

Formulas follow the survey (which cites the original paper for each function; that paper is given
in `src`). Descriptions are our own. Where a function has free parameters the card plots the stated
example values (`note`). Ranges and smoothness left as None are computed by build.py from the curve.

Same fields as catalog.py; this module is imported at the end of catalog.py.
"""
import math

from catalog import fn, sig, softplus, Phi, relu, gelu, silu

SURVEY = "via Kunc & Kléma, 2024"


def sech(x):
    return 0.0 if abs(x) > 700 else 1 / math.cosh(x)


def sinc(x):
    return 1.0 if x == 0 else math.sin(x) / x


def S(src):
    """Original citation plus the survey it was collected from."""
    return f"{src} ({SURVEY})"


# =============================================================================== batch 1
# Sigmoid family (survey §1.2) and sigmoid-weighted linear units (survey §1.3)

# ---- sigmoid variants
fn("shifted-scaled-sigmoid", "Shifted and scaled sigmoid", "sigmoid-variants", lambda x: sig(2 * (x - 1)),
   "f(x) = σ(a(x − b))",
   "The logistic sigmoid stretched by a and shifted by b, so the transition can be placed and sharpened for a particular input scale.",
   None, True, aka="SSS", src=S("Arai & Imamura, 2018"), note="a = 2, b = 1")
fn("variant-sigmoid", "Variant sigmoid function", "sigmoid-variants", lambda x: 2 * sig(x) - 1,
   "f(x) = a σ(bx) − c",
   "An early three-parameter sigmoid: a scales the output, b sets the slope and c shifts it down, which can make it zero-centred.",
   None, True, aka="VSF", src=S("Han & Moraga, 1995"), note="a = 2, b = 1, c = 1")
fn("bimodal-sigmoid", "Bi-modal sigmoid", "sigmoid-variants", lambda x: 0.5 * (sig(x) + sig(x + 4)),
   "f(x) = ½ [σ(x) + σ(x + b)]",
   "The average of two shifted sigmoids, so the curve rises in two steps and its derivative has two peaks.",
   None, True, src=S("Sodhi & Chandra, 2014"), note="b = 4", xr=(-8, 4))
fn("sigmoid-algebraic", "Sigmoid-Algebraic", "sigmoid-variants",
   lambda x: sig(x * (1 + abs(x)) / (1 + abs(x) * (1 + abs(x)))),
   "f(x) = σ( x(1 + a|x|) / (1 + |x|(1 + a|x|)) )",
   "A sigmoid applied to an algebraic squashing of the input. Because the inner term is itself bounded, the output never gets close to 0 or 1.",
   None, None, src=S("Koçak & Üstündağ Şiray, 2021"), note="a = 1")
fn("triple-state-sigmoid", "Triple-state sigmoid", "sigmoid-variants",
   lambda x: sig(x) * (sig(x) + sig(x - 2) + sig(x - 4)),
   "f(x) = σ(x) [σ(x) + σ(x − a) + σ(x − b)]",
   "A sigmoid multiplied by a sum of three shifted sigmoids, so it climbs to 3 through a staircase of intermediate levels.",
   None, True, aka="TS-sigmoid", src=S("Koçak & Üstündağ Şiray, 2021"), note="a = 2, b = 4", xr=(-4, 8))
fn("improved-logistic-sigmoid", "Improved logistic sigmoid", "sigmoid-variants",
   lambda x: 0.2 * (x - 2) + sig(2) if x >= 2 else (0.2 * (x + 2) + sig(-2) if x <= -2 else sig(x)),
   [("a(x − b) + σ(b)", "x ≥ b"), ("σ(x)", "−b < x < b"), ("a(x + b) + σ(−b)", "x ≤ −b")],
   "A sigmoid in the middle with straight lines of slope a outside ±b, so the gradient never vanishes. The slope a should be larger than the sigmoid's own slope at b.",
   None, None, src=S("Qin et al., 2019"), note="a = 0.2, b = 2; outer pieces joined continuously", xr=(-6, 6))
fn("siglin", "SigLin", "sigmoid-variants", lambda x: sig(x) + 0.1 * x, "f(x) = σ(x) + ax",
   "A sigmoid plus a small linear term, which keeps a non-zero gradient everywhere.",
   None, True, aka="Sigmoid + linear", src=S("Roodschild et al., 2020"), note="a = 0.1", xr=(-8, 8))
fn("soft-root-sign", "Soft-root-sign", "sigmoid-variants", lambda x: x / (x / 2 + math.exp(-x / 3)) if x > -300 else 0.0,
   "f(x) = x / (x/a + e<sup>−x/b</sup>)",
   "Smooth, non-monotonic and bounded: positive inputs saturate at a, while negative inputs dip below zero and then return to it.",
   None, True, aka="SRS", src=S("Li & Zhou, 2020"), note="a = 2, b = 3", xr=(-6, 10))
fn("soft-clipping", "Soft clipping", "sigmoid-variants", lambda x: (softplus(5 * x) - softplus(5 * (x - 1))) / 5,
   "f(x) = (1/a) ln[(1 + e<sup>ax</sup>) / (1 + e<sup>a(x − 1)</sup>)]",
   "A smooth version of clipping to [0, 1]: nearly linear between 0 and 1 and flat outside, with a controlling how sharp the corners are.",
   None, True, aka="SC", src=S("Klimek & Perelstein, 2020"), note="a = 5", xr=(-2, 3))
fn("hexpo", "Hexpo", "sigmoid-variants", lambda x: -math.expm1(-x) if x >= 0 else math.expm1(x),
   [("−a(e<sup>−x/b</sup> − 1)", "x ≥ 0"), ("c(e<sup>x/d</sup> − 1)", "x < 0")],
   "Two exponential pieces that saturate at a and −c, giving a tanh-like curve whose gradient can be scaled separately on each side.",
   None, True, src=S("Kong & Takatsuka, 2017"),
   note="a = b = c = d = 1; negative branch written so it saturates, as the original intends")
fn("smooth-step", "Smooth step", "sigmoid-variants",
   lambda x: 1.0 if x >= 2 else (0.0 if x <= -2 else -2 / 64 * x ** 3 + 3 / 8 * x + 0.5),
   [("1", "x ≥ a/2"), ("−(2/a³)x³ + (3/2a)x + ½", "|x| < a/2"), ("0", "x ≤ −a/2")],
   "A cubic sigmoid that is exactly 0 below −a/2 and exactly 1 above a/2, so it can switch paths off completely. Used for conditional computation in differentiable tree ensembles.",
   None, True, src=S("Hazimeh et al., 2020"), note="a = 4")
fn("elliott", "Elliott function", "sigmoid-variants", lambda x: 0.5 * x / (1 + abs(x)) + 0.5,
   "f(x) = 0.5x / (1 + |x|) + 0.5",
   "Softsign rescaled to (0, 1): a sigmoid substitute that needs no exponential, which mattered on early hardware.",
   "(0, 1)", True, src=S("Elliott, 1993"), xr=(-8, 8))
fn("sinc-sigmoid", "Sinc-Sigmoid", "sigmoid-variants", lambda x: sinc(sig(x)),
   "f(x) = sinc(σ(x)) = sin(σ(x)) / σ(x)",
   "The unnormalised sinc applied to a sigmoid. Since σ(x) stays in (0, 1), the result is a gentle decreasing curve from 1 toward sin(1).",
   None, True, src=S("Koçak & Üstündağ Şiray, 2021"))
fn("sigmoid-gumbel", "Sigmoid-Gumbel", "sigmoid-variants", lambda x: sig(x) * math.exp(-math.exp(-x)) if x > -50 else 0.0,
   "f(x) = σ(x) · exp(−e<sup>−x</sup>)",
   "The product of the logistic and Gumbel CDFs: a sigmoid that rises later and more steeply than σ(x).",
   None, True, aka="SG", src=S("Kaytan et al., 2022"))
fn("new-sigmoid", "NewSigmoid", "sigmoid-variants",
   lambda x: math.copysign(1 / math.sqrt(2), x) if abs(x) > 20 else math.sinh(x) / math.sqrt(math.cosh(2 * x)),
   "f(x) = (e<sup>x</sup> − e<sup>−x</sup>) / √(2(e<sup>2x</sup> + e<sup>−2x</sup>))",
   "A tanh-like curve that saturates at ±1/√2.",
   None, True, src=S("Kumar & Sodhi, 2023"))
fn("root2sigmoid", "Root2sigmoid", "sigmoid-variants",
   lambda x: math.copysign(0.25, x) if abs(x) > 200 else
   (math.sqrt(2) ** x - math.sqrt(2) ** -x) / (2 * math.sqrt(2) * math.sqrt(2 * (2 ** x + 2 ** -x))),
   "f(x) = (√2<sup>x</sup> − √2<sup>−x</sup>) / (2√2 · √(2(√2<sup>2x</sup> + √2<sup>−2x</sup>)))",
   "A base-√2 relative of NewSigmoid that saturates at ±1/4. The survey notes the original paper's formula appears to contain a typo; this is the survey's corrected form.",
   None, True, src=S("Kumar & Sodhi, 2023"), xr=(-10, 10))
fn("loglog", "LogLog", "sigmoid-variants", lambda x: math.exp(-math.exp(-x)) if x > -50 else 0.0,
   "f(x) = exp(−e<sup>−x</sup>)",
   "The Gumbel CDF used as a sigmoid: asymmetric, leaving 0 quickly and approaching 1 slowly (the mirror image of cloglog). Used in networks for financial time-series forecasting.",
   "(0, 1)", True, src=S("Gomes et al., 2010"))
fn("sechsig", "SechSig", "sigmoid-variants", lambda x: (x + sech(x)) * sig(x),
   "f(x) = (x + sech(x)) · σ(x)",
   "A Swish-like gated unit whose gated input gets a bump from sech(x) near zero.",
   None, True, src=S("Közkurt et al., 2023"))
fn("psechsig", "Parametric SechSig", "sigmoid-variants", lambda x: (x + 0.5 * sech(x + 0.5)) * sig(x),
   "f(x) = (x + a · sech(x + a)) · σ(x)",
   "SechSig with a fixed parameter a that sets both the height and the position of the bump.",
   None, True, aka="pSechSig", src=S("Közkurt et al., 2023"), note="a = 0.5")
fn("tanhsig", "TanhSig", "sigmoid-variants", lambda x: (x + math.tanh(x)) * sig(x),
   "f(x) = (x + tanh(x)) · σ(x)",
   "Like SechSig but adds tanh(x) to the gated input, steepening the curve around zero.",
   None, True, src=S("Közkurt et al., 2023"))
fn("ptanhsig", "Parametric TanhSig", "sigmoid-variants", lambda x: (x + 0.5 * math.tanh(x + 0.5)) * sig(x),
   "f(x) = (x + a · tanh(x + a)) · σ(x)",
   "TanhSig with a fixed parameter a that scales and shifts the tanh term.",
   None, True, aka="pTanhSig", src=S("Közkurt et al., 2023"), note="a = 0.5")
fn("msaf", "Multistate activation function", "sigmoid-variants", lambda x: -1 + sig(x) + sig(x + 4),
   "f(x) = a + Σ<sub>k</sub> σ(x − b<sub>k</sub>)",
   "Sums shifted sigmoids so a neuron has several stable output levels instead of two. Plotted as the two-step example −1 + σ(x) + σ(x + 4).",
   None, True, aka="MSAF", src=S("Cai et al., 2015"), xr=(-9, 5))
fn("rootsig", "Rootsig", "sigmoid-variants", lambda x: x / (1 + math.sqrt(1 + x * x)),
   "f(x) = ax / (1 + √(1 + a²x²))",
   "An algebraic sigmoid that saturates at ±1 without exponentials, approaching its limits polynomially.",
   None, True, src=S("Duch & Jankowski, 1999"), note="a = 1", xr=(-10, 10))
fn("sigmoid-tanh-combinations", "Sigmoid–tanh combinations", "sigmoid-variants",
   [("{σ₂, tanh}", lambda x: 2 * sig(x) - 1 if x >= 0 else math.tanh(x)),
    ("{tanh, 0}", lambda x: math.tanh(x) if x >= 0 else 0.0),
    ("{σ₂, ax}", lambda x: 2 * sig(x) - 1 if x >= 0 else 0.1 * x)],
   [("g(x)", "x ≥ 0"), ("h(x)", "x < 0")],
   "A family that uses one function for positive inputs and another for negative ones, mixing the bipolar sigmoid σ₂(x) = 2σ(x) − 1, tanh, zero and a linear piece. Three of the proposed pairs are plotted.",
   "depends on the pair", False, src=S("Guevraa et al., 2021"), note="a = 0.1", mono=True)

# ---- sigmoid-weighted linear units (x · gate)
fn("sgelu", "Symmetrical GELU", "swish-descendants", lambda x: x * math.erf(x / math.sqrt(2)),
   "f(x) = a · x · erf(x / √2)",
   "A symmetric (even) variant of GELU: negative inputs give positive outputs, mirroring the positive side. Its authors report more symmetrically distributed weights.",
   None, True, aka="SGELU", src=S("Yu & Su, 2019"), note="a = 1", xr=(-3, 3))
fn("calu", "Cauchy linear unit", "swish-descendants", lambda x: x * (math.atan(x) / math.pi + 0.5),
   "f(x) = x · (arctan(x)/π + ½)",
   "GELU's idea with the Cauchy CDF as the gate. The heavy-tailed gate opens slowly, so the negative lobe is wide and keeps growing in magnitude.",
   None, True, aka="CaLU", src=S("Wu et al., 2023"))
fn("lalu", "Laplace linear unit", "swish-descendants",
   lambda x: x * (1 - 0.5 * math.exp(-x)) if x >= 0 else x * 0.5 * math.exp(x),
   "f(x) = x · Φ<sub>Laplace</sub>(x)",
   "GELU's idea with the Laplace CDF as the gate, which has exponential tails like the logistic but a sharper peak in its density.",
   None, True, aka="LaLU", src=S("Wu et al., 2023"))
fn("colu", "Collapsing linear unit", "swish-descendants", lambda x: x / (1 - x * math.exp(-(x + math.exp(x)))) if x > -700 else 0.0,
   "f(x) = x / (1 − x e<sup>−(x + e<sup>x</sup>)</sup>)",
   "A self-gated unit whose gate is built from a double exponential, collapsing negative inputs toward zero.",
   None, True, aka="CoLU", src=S("Vagerwal, 2021"))
fn("triple-state-swish", "Triple-state swish", "swish-descendants",
   lambda x: x * sig(x) * (sig(x) + sig(x - 2) + sig(x - 4)),
   "f(x) = x σ(x) [σ(x) + σ(x − a) + σ(x − b)]",
   "SiLU multiplied by a sum of three shifted sigmoids, so the slope increases in stages as the input grows.",
   None, True, aka="TS-swish", src=S("Koçak & Üstündağ Şiray, 2021"), note="a = 2, b = 4", xr=(-4, 6))
fn("generalized-swish", "Generalised swish", "swish-descendants", lambda x: x * sig(math.exp(-x)) if x > -700 else x,
   "f(x) = x · σ(e<sup>−x</sup>)",
   "Gates the input with the sigmoid of e<sup>−x</sup>: the gate is about 1 for negative inputs and falls to ½ for large positive ones.",
   None, True, src=S("Koçak & Üstündağ Şiray, 2021"))
fn("exponential-swish", "Exponential swish", "swish-descendants", lambda x: math.exp(-x) * sig(x) if x > -700 else 1.0,
   "f(x) = e<sup>−x</sup> σ(x)",
   "Despite the name, this simplifies algebraically to σ(−x): a decreasing sigmoid.",
   None, True, src=S("Koçak & Üstündağ Şiray, 2021"))
fn("sigmoid-derivative", "Derivative of sigmoid", "swish-descendants", lambda x: sig(x) * (1 - sig(x)),
   "f(x) = e<sup>−x</sup> σ(x)² = σ(x)(1 − σ(x))",
   "The sigmoid's own derivative used as an activation: a bell-shaped bump peaking at 0.25, similar to a Gaussian.",
   None, True, src=S("Koçak & Üstündağ Şiray, 2021"))
fn("gish", "Gish", "swish-descendants", lambda x: x * math.log(2 - math.exp(-math.exp(x))) if x < 700 else x * math.log(2),
   "f(x) = x · ln(2 − exp(−e<sup>x</sup>))",
   "A self-gated unit whose gate rises from 0 to ln 2. Its authors report it beating Mish, Logish and Smish on MNIST and CIFAR-10.",
   None, True, src=S("Kaytan et al., 2023"))
fn("loglogish", "LogLogish", "swish-descendants", lambda x: x * -math.expm1(-math.exp(x)) if x < 700 else x,
   "f(x) = x · (1 − exp(−e<sup>x</sup>))",
   "x gated by the complementary log-log curve, which opens late and then quickly.",
   None, True, src=S("Wu et al., 2023"))
fn("expexpish", "ExpExpish", "swish-descendants", lambda x: x * math.exp(-math.exp(-x)) if x > -50 else 0.0,
   "f(x) = x · exp(−e<sup>−x</sup>)",
   "x gated by the LogLog (Gumbel CDF) curve, which opens early and then slowly.",
   None, True, src=S("Wu et al., 2023"))
fn("self-arctan", "Self arctan", "swish-descendants", lambda x: x * math.atan(x),
   "f(x) = x · arctan(x)",
   "Even and non-negative, like LiSHT but built from arctan: both large positive and large negative inputs give large positive outputs.",
   None, True, src=S("Tümer-Sivri et al., 2023"), xr=(-3, 3))
fn("parametric-logish", "Parametric Logish", "swish-descendants", lambda x: x * math.log1p(sig(10 * x)),
   "f(x) = a x · ln(1 + σ(bx))",
   "Logish with fixed scales a and b; a large b makes the gate switch sharply at zero.",
   None, True, aka="pLogish", src=S("Zhu et al., 2021"), note="a = 1, b = 10", xr=(-3, 3))
fn("phish", "Phish", "swish-descendants", lambda x: x * math.tanh(gelu(x)),
   "f(x) = x · tanh(GELU(x))",
   "Gates x with tanh of GELU. Its author found it performed on a par with Mish and Swish.",
   None, True, src=S("Naveen, 2022"))


# =============================================================================== batch 2
# More sigmoid-weighted units (survey §1.3) and the first fixed ReLU variants (survey §1.6)

fn("suish", "Suish", "swish-descendants", lambda x: max(x, x * math.exp(-abs(x))),
   "f(x) = max(x, x · e<sup>−|x|</sup>)",
   "Identity for positive inputs; for negative inputs it follows x·e^{x}, which dips and then returns toward zero, like a sharper Swish.",
   None, None, src=S("Xu & Philip Chen, 2020"))
fn("tsrelu", "Tangent-sigmoid ReLU", "swish-descendants", lambda x: x * math.tanh(sig(x)),
   "f(x) = x · tanh(σ(x))",
   "Mish's recipe with the sigmoid inside tanh. Because tanh(σ(x)) never exceeds tanh(1) ≈ 0.76, the positive slope is below 1.",
   None, True, aka="TSReLU", src=S("Mercioni et al., 2020"))
fn("tbsrelu", "Tangent-bipolar-sigmoid ReLU", "swish-descendants", lambda x: x * math.tanh(math.tanh(x / 2)),
   "f(x) = x · tanh((1 − e<sup>−x</sup>) / (1 + e<sup>−x</sup>))",
   "TSReLU with the bipolar sigmoid tanh(x/2) inside, so the gate is negative for negative inputs and the output is non-negative on both sides.",
   None, True, aka="TBSReLU", src=S("Mercioni et al., 2020"))
fn("dsilu", "dSiLU", "swish-descendants", lambda x: sig(x) * (1 + x * (1 - sig(x))),
   "f(x) = σ(x) (1 + x(1 − σ(x)))",
   "The derivative of SiLU used as an activation: a sigmoid-like curve that overshoots to about 1.1 and dips to about −0.1. Proposed for reinforcement-learning value networks.",
   None, True, aka="Derivative of SiLU", src=S("Elfwing et al., 2018"), xr=(-8, 8))
fn("doublesilu", "DoubleSiLU", "swish-descendants", lambda x: x * sig(x * sig(x)),
   "f(x) = x · σ(x · σ(x))",
   "SiLU whose gate is itself driven by SiLU, which flattens the negative lobe.",
   None, True, src=S("Verma et al., 2023"))
fn("msilu", "Modified SiLU", "swish-descendants", lambda x: x * sig(x) + math.exp(-x * x - 1) / 4,
   "f(x) = x σ(x) + e<sup>−x² − 1</sup> / 4",
   "SiLU plus a small Gaussian bump at the origin, which its authors found speeds up convergence.",
   None, True, aka="MSiLU", src=S("Hayou et al., 2018"))
fn("tsilu", "Tanh SiLU", "swish-descendants", lambda x: math.tanh(x * sig(x)),
   "f(x) = tanh(x · σ(x))",
   "SiLU squashed by tanh, which bounds the output at 1. (The survey's printed formula has a typo in the denominator; this is the intended function.)",
   None, True, aka="TSiLU", src=S("Verma et al., 2023"))
fn("asilu", "Arctan SiLU", "swish-descendants", lambda x: math.atan(x * sig(x)),
   "f(x) = arctan(x · σ(x))",
   "SiLU passed through arctan, which bounds it at π/2 but approaches the bound slowly.",
   None, True, aka="ASiLU", src=S("Verma et al., 2023"))
fn("swat", "SwAT", "swish-descendants", lambda x: x * sig(math.atan(x)),
   "f(x) = x · σ(arctan(x))",
   "x gated by the sigmoid of arctan(x). The gate stays between σ(−π/2) ≈ 0.17 and σ(π/2) ≈ 0.83, so both sides grow linearly.",
   None, True, src=S("Verma et al., 2023"))
fn("rectified-sech", "Rectified hyperbolic secant", "swish-descendants", lambda x: x * sech(x),
   "f(x) = x · sech(x)",
   "Odd and smooth, peaking near x ≈ 1.2 and decaying back to zero at both ends, so large inputs of either sign are suppressed.",
   None, True, src=S("Njikam & Zhao, 2016"), xr=(-6, 6))
fn("eanaf", "EANAF", "swish-descendants", lambda x: x * math.exp(x) / (math.exp(x) + 2) if x < 700 else x,
   "f(x) = x e<sup>x</sup> / (e<sup>x</sup> + 2)",
   "Efficient asymmetric nonlinear activation: x · tanh(softplus(x)/2) simplified to a cheap closed form. Very close to Swish.",
   None, True, src=S("Chai et al., 2022"))
fn("sinsig", "SinSig", "swish-descendants", lambda x: x * math.sin(math.pi / 2 * sig(x)),
   "f(x) = x · sin(π/2 · σ(x))",
   "A self-gated unit whose gate is a sine of the sigmoid. Its authors report it scaling better than Swish and Mish as networks get deeper.",
   None, True, src=S("Douge et al., 2021"))
fn("sielu", "SiELU", "swish-descendants",
   lambda x: x * sig(2 * math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)),
   "f(x) = x · σ(2√(2/π) (x + 0.044715x³))",
   "The tanh approximation of GELU rewritten with a sigmoid (since tanh(u) = 2σ(2u) − 1), so it equals GELU's tanh form exactly.",
   None, True, src=S("Ahmed et al., 2023"))

# ---- fixed ReLU variants
fn("shifted-relu", "Shifted ReLU", "relu-variants", lambda x: max(-1.0, x), "f(x) = max(−1, x)",
   "ReLU moved down by one so the floor is at −1, which brings mean activations closer to zero.",
   None, False, src=S("Clevert et al., 2016"))
fn("sloped-relu", "Sloped ReLU", "relu-variants", lambda x: 2 * x if x >= 0 else 0.0,
   [("ax", "x ≥ 0"), ("0", "x < 0")],
   "ReLU with a steeper positive slope a (its authors recommend 1 to 10), the mirror image of Leaky ReLU's idea.",
   None, False, aka="SlReLU", src=S("Seo et al., 2017"), note="a = 2")
fn("sine-relu", "SineReLU", "relu-variants", lambda x: x if x >= 0 else 0.5 * (math.sin(x) - math.cos(x)),
   [("x", "x ≥ 0"), ("a(sin x − cos x)", "x < 0")],
   "Identity for positive inputs and a small oscillation for negative ones, so negative inputs keep a non-zero gradient.",
   None, False, src=S("Gupta & Ahuja, 2021; Rodrigues, 2018"), note="a = 0.5", xr=(-8, 4))
fn("minsin", "Minsin", "relu-variants", lambda x: min(x, math.sin(x)),
   "f(x) = min(x, sin(x))",
   "The identity for negative inputs and a sine for positive ones, so positive outputs stay bounded by 1.",
   None, True, src=S("Xu & Philip Chen, 2020"), xr=(-6, 10))
fn("vlu", "Variational linear unit", "relu-variants", lambda x: relu(x) + 0.25 * math.sin(4 * x),
   "f(x) = max(0, x) + a sin(bx)",
   "ReLU with a sinusoidal ripple added everywhere, so even negative inputs have a gradient.",
   None, False, aka="VLU", src=S("Gupta & Ahuja, 2021"), note="a = 0.25, b = 4")
fn("nlrelu", "Natural-logarithm ReLU", "relu-variants", lambda x: math.log(1 * relu(x) + 1),
   "f(x) = ln(a · max(0, x) + 1)",
   "ReLU with a logarithmic positive side, which compresses large activations.",
   None, False, aka="NLReLU", src=S("Liu et al., 2019"), note="a = 1", xr=(-3, 8))
fn("slu", "Softplus linear unit", "relu-variants", lambda x: x if x >= 0 else 2 * math.log((math.exp(x) + 1) / 2),
   [("x", "x ≥ 0"), ("2 ln((e<sup>x</sup> + 1) / 2)", "x < 0")],
   "Identity for positive inputs and a scaled, shifted softplus for negative ones, with parameters chosen so the function is continuous and differentiable at zero and negative outputs pull the mean toward zero.",
   None, True, aka="SLU", src=S("Zhao et al., 2017"))
fn("resp", "Rectified softplus", "relu-variants", lambda x: 1.4 * x + math.log(2) if x >= 0 else softplus(x),
   [("ax + ln 2", "x ≥ 0"), ("ln(1 + e<sup>x</sup>)", "x < 0")],
   "Softplus for negative inputs and a steeper line for positive ones; slopes between 1.4 and 2.0 were found to work well.",
   None, False, aka="ReSP", src=S("Xu et al., 2018"), note="a = 1.4")
fn("prenu", "Parametric rectified non-linear unit", "relu-variants", lambda x: x - 0.5 * math.log(x + 1) if x >= 0 else 0.0,
   [("x − a ln(x + 1)", "x ≥ 0"), ("0", "x < 0")],
   "ReLU whose positive side bends slightly below the identity, making the positive part nonlinear.",
   None, None, aka="PReNU", src=S("Jaafari et al., 2020"), note="a = 0.5", xr=(-3, 6))
fn("bounded-relu", "Bounded ReLU", "relu-variants", lambda x: min(max(0.0, x), 2.0),
   "f(x) = min(max(0, x), a)",
   "ReLU capped at a, which prevents unbounded activations; ReLU6 is the special case a = 6. Also published independently as ReLUN.",
   None, False, aka="BReLU, ReLUN", src=S("Liew et al., 2016"), note="a = 2")
fn("sv-hardtanh", "Vertically shifted HardTanh", "relu-variants", lambda x: max(-1.0, min(1.0, x)) + 0.5,
   "f(x) = HardTanh(x) + a",
   "HardTanh moved up by a, so its output is centred at a instead of zero.",
   None, False, aka="SvHardTanh", src=S("Kim et al., 2021"), note="a = 0.5")
fn("sh-hardtanh", "Horizontally shifted HardTanh", "relu-variants", lambda x: max(-1.0, min(1.0, x + 0.5)),
   [("−1", "x < −1 − a"), ("x + a", "−1 − a ≤ x ≤ 1 − a"), ("1", "x > 1 − a")],
   "HardTanh moved left by a, so its linear zone is no longer centred on zero.",
   None, False, aka="ShHardTanh", src=S("Kim et al., 2021"),
   note="a = 0.5; middle piece written as x + a so the function is continuous")


# =============================================================================== batch 3
# Remaining fixed ReLU / ELU variants (survey §1.6–1.7) and square-based functions (survey §1.8)
SELU_A, SELU_B = 1.0507009873554805, 1.6732632423543772

fn("blrelu", "Bounded leaky ReLU", "relu-variants",
   lambda x: 0.1 * x if x <= 0 else (x if x < 2 else 0.1 * x + 1.8),
   [("ax", "x ≤ 0"), ("x", "0 < x < b"), ("ax + c", "x > b")],
   "Leaky ReLU whose positive side also flattens to slope a above the threshold b, with c = (1 − a)b keeping it continuous.",
   None, False, aka="BLReLU", src=S("Liew et al., 2016"), note="a = 0.1, b = 2", xr=(-4, 6))
fn("pan-function", "Pan function", "relu-variants",
   lambda x: x - 1 if x >= 1 else (-x - 1 if x <= -1 else 0.0),
   [("x − a", "x ≥ a"), ("0", "−a < x < a"), ("−x − a", "x ≤ −a")],
   "Zero inside [−a, a] and rising linearly outside it on both sides: soft thresholding of the absolute value.",
   None, False, src=S("Pan et al., 2023"), note="a = 1")
fn("abslu", "Absolute linear unit", "relu-variants", lambda x: x if x >= 0 else 0.5 * abs(x),
   [("x", "x ≥ 0"), ("a|x|", "x < 0")],
   "Identity for positive inputs and a scaled absolute value for negative ones, so negative inputs also produce positive outputs.",
   None, False, aka="AbsLU", src=S("Alkhouly et al., 2021"), note="a = 0.5")
fn("mrelu", "Mirrored ReLU", "relu-variants", lambda x: min(relu(1 - x), relu(1 + x)),
   "f(x) = min(ReLU(1 − x), ReLU(1 + x))",
   "A triangular 'tent' that peaks at 1 for x = 0 and is zero outside [−1, 1], suppressing unusual (large) inputs.",
   None, False, aka="mReLU", src=S("Zhao & Griffin, 2016"), xr=(-3, 3))
fn("lsptlu", "Leaky single-peaked triangle linear unit", "relu-variants",
   lambda x: 0.2 * x if x < 0 else (x if x <= 1 else (2 - x if x <= 2 else 0.0)),
   [("0.2x", "x < 0"), ("x", "0 ≤ x ≤ a"), ("2a − x", "a < x ≤ 2a"), ("0", "x > 2a")],
   "A leaky ReLU that peaks at a and falls back to zero by 2a, so only a band of positive inputs is passed strongly.",
   None, False, aka="LSPTLU, LRTLU", src=S("Shan et al., 2019"), note="a = 1", xr=(-3, 4))
fn("softmodulusq", "SoftModulusQ", "relu-variants", lambda x: x * x * (2 - abs(x)) if abs(x) <= 1 else abs(x),
   [("x²(2 − |x|)", "|x| ≤ 1"), ("|x|", "|x| > 1")],
   "A smooth stand-in for |x|: a quadratic-like patch near zero that joins the absolute value at ±1.",
   None, True, src=S("Vallés-Pérez et al., 2023"), note="conditions written as |x| ≤ 1 / |x| > 1 (the survey's printed conditions overlap)", xr=(-3, 3))
fn("softmodulust", "SoftModulusT", "relu-variants", lambda x: x * math.tanh(x / 0.5),
   "f(x) = x · tanh(x / a)",
   "A tanh-smoothed absolute value; a = 1 gives LiSHT and smaller a makes it closer to |x|. Its authors used a = 0.01.",
   None, True, src=S("Vallés-Pérez et al., 2023"), note="a = 0.5", xr=(-3, 3))
fn("signrelu", "SignReLU", "relu-variants", lambda x: x if x >= 0 else x / (abs(x) + 1),
   [("x", "x ≥ 0"), ("a x / (|x| + 1)", "x < 0")],
   "ReLU for positive inputs and a softsign for negative ones, which saturates at −a instead of being zero. Also published as DLU.",
   None, True, aka="DLU", src=S("Lin & Shen, 2018"), note="a = 1", xr=(-8, 4))
fn("li-relu", "Li-ReLU", "relu-variants", lambda x: 0.2 * x + x if x >= 0 else 0.2 * x,
   [("ax + x", "x ≥ 0"), ("ax", "x < 0")],
   "ReLU plus a linear term ax, equivalent to a leaky ReLU whose positive slope is 1 + a.",
   None, False, src=S("Elakkiya & Dejey, 2024"), note="a = 0.2")
fn("ncrelu", "Negative CReLU", "relu-variants", [("ReLU(x)", relu), ("−ReLU(−x)", lambda x: -relu(-x))],
   "f(x) = [ReLU(x), −ReLU(−x)]",
   "Like CReLU, it doubles each channel, but keeps the negative part's sign, so the two outputs are max(x, 0) and min(x, 0).",
   "ℝ²", False, aka="NCReLU", src=S("Zagoruyko & Komodakis, 2017"), mono=True)
fn("repu", "Rectified power unit", "algebraic",
   [("a = 0.5", lambda x: x ** 0.5 if x >= 0 else 0.0), ("a = 2", lambda x: x ** 2 if x >= 0 else 0.0),
    ("a = 3", lambda x: x ** 3 if x >= 0 else 0.0)],
   [("x<sup>a</sup>", "x ≥ 0"), ("0", "x < 0")],
   "ReLU raised to a power a: a = 0 gives the step function, a = 1 ReLU, a = 2 the squared ReLU. Also proposed earlier as a 'power activation'.",
   "[0, ∞)", None, aka="RePU, power activation", src=S("Berradi, 2018; Li et al., 2019"), xr=(-1, 2), mono=True)
fn("apprelu", "Approximate ReLU", "algebraic", lambda x: 0.5 * x ** 1.5 if x >= 0 else 0.0,
   [("a x<sup>b</sup>", "x ≥ 0"), ("0", "x < 0")],
   "RePU with an extra scale a.",
   None, None, aka="AppReLU", src=S("Saha et al., 2020"), note="a = 0.5, b = 1.5", xr=(-2, 4))
fn("eplaf", "Even power-linear (EPLAF)", "algebraic",
   lambda x: x - 0.5 if x >= 1 else (abs(x) ** 2 / 2 if x >= -1 else -x - 0.5),
   [("x − (1 − 1/d)", "x ≥ 1"), ("(1/d)|x|<sup>d</sup>", "−1 ≤ x < 1"), ("−x − (1 − 1/d)", "x < −1")],
   "An even function: a power curve near zero joined to straight lines, so it grows linearly on both sides.",
   None, True, aka="EvenPowLin", src=S("Nasiri & Ghiasi-Shirazi, 2021"), note="d = 2")
fn("oplaf", "Odd power-linear (OPLAF)", "algebraic",
   lambda x: x - 0.5 if x >= 1 else (x * x / 2 if x >= 0 else (-x * x / 2 if x >= -1 else -x - 0.5)),
   [("x − (1 − 1/d)", "x ≥ 1"), ("(1/d)|x|<sup>d</sup>", "0 ≤ x < 1"), ("−(1/d)|x|<sup>d</sup>", "−1 ≤ x < 0"),
    ("−x − (1 − 1/d)", "x < −1")],
   "The odd counterpart of EPLAF, as printed in the survey: a power curve near zero (negated for negative inputs) joined to straight lines. With these outer pieces both sides end up rising linearly.",
   None, None, aka="OddPowLin", src=S("Nasiri & Ghiasi-Shirazi, 2021"), note="d = 2", mono=False)
fn("drlu", "Delay ReLU", "relu-variants", lambda x: relu(x - 0.5),
   "f(x) = max(0, x − a)",
   "ReLU shifted right by a fixed delay a, so it only switches on once the input exceeds a.",
   None, False, aka="DRLU", src=S("Shan et al., 2022"), note="a = 0.5")
fn("disrelu", "Displaced ReLU", "relu-variants", lambda x: x if x + 0.5 >= 0 else -0.5,
   [("x", "x + a ≥ 0"), ("−a", "x + a < 0")],
   "ReLU whose corner is moved to (−a, −a); a = 1 gives the shifted ReLU.",
   None, False, aka="DisReLU", src=S("Macêdo et al., 2019"), note="a = 0.5")
fn("mlrelu", "Modified leaky ReLU", "relu-variants", lambda x: x if x + 0.5 > 0 else -0.5 * x,
   [("x", "x + a > 0"), ("−ax", "x + a ≤ 0")],
   "A translated leaky ReLU in which one parameter a sets both the threshold and the negative slope. As written, the function jumps at x = −a.",
   None, False, aka="MLReLU", src=S("Yang et al., 2022"), note="a = 0.5", mono=False)
fn("flatted-t-swish", "Flatted-T swish", "relu-variants", lambda x: relu(x) * sig(x) - 0.2,
   "f(x) = ReLU(x) · σ(x) + T",
   "Swish on the positive side and a flat value T on the negative side (T = −0.2 is recommended); T = 0 was also published as ReLU-Swish.",
   None, True, aka="FTS", src=S("Chieng et al., 2018"), note="T = −0.2")
fn("oaf", "Optimal activation function", "relu-variants", lambda x: relu(x) + x * sig(x),
   "f(x) = ReLU(x) + x · σ(x)",
   "The sum of ReLU and SiLU: Swish's negative lobe with a steeper positive slope.",
   None, False, aka="OAF", src=S("Sharma, 2020"))
fn("reu", "Rectified exponential unit", "relu-variants", lambda x: x if x >= 0 else x * math.exp(x),
   [("x", "x ≥ 0"), ("x e<sup>x</sup>", "x < 0")],
   "Identity for positive inputs and x·eˣ for negative ones, which gives Swish's non-monotonic dip with ReLU's positive side.",
   None, True, aka="REU", src=S("Ying et al., 2019"))
fn("ada", "Apical dendrite activation", "relu-variants", lambda x: math.exp(-x) if x >= 0 else 0.0,
   [("exp(−ax + b)", "x ≥ 0"), ("0", "x < 0")],
   "A biologically inspired unit that fires most strongly just above zero and decays for larger inputs, like an apical dendrite's response.",
   None, False, aka="ADA", src=S("Georgescu et al., 2023"), note="a = 1, b = 0", mono=False)
fn("lada", "Leaky apical dendrite activation", "relu-variants", lambda x: math.exp(-x) if x >= 0 else 0.01 * x,
   [("exp(−ax + b)", "x ≥ 0"), ("cx", "x < 0")],
   "ADA with a small leaky slope c for negative inputs.",
   None, False, aka="LADA", src=S("Georgescu et al., 2023"), note="a = 1, b = 0, c = 0.01", mono=False)
fn("siglu", "SigLU", "relu-variants", lambda x: x if x >= 0 else math.tanh(x),
   [("x", "x ≥ 0"), ("(1 − e<sup>−2x</sup>) / (1 + e<sup>−2x</sup>)", "x < 0")],
   "An ELU alternative whose negative side is tanh(x), saturating at −1.",
   None, True, src=S("Guevraa et al., 2021"))
fn("sara", "SaRa", "relu-variants", lambda x: x if x >= 0 else x / (1 + 0.5 * math.exp(-0.7 * x)),
   [("x", "x ≥ 0"), ("x / (1 + a e<sup>−bx</sup>)", "x < 0")],
   "Swish-like on the negative side and identity on the positive side. (The survey reconstructs the formula; the original paper's version was malformed.)",
   None, None, aka="Swish and ReLU activation", src=S("Qureshi & Umar, 2022"), note="a = 0.5, b = 0.7")
fn("maxsig", "Maxsig", "relu-variants", lambda x: max(x, sig(x)),
   "f(x) = max(x, σ(x))",
   "The larger of x and the sigmoid: sigmoid-shaped for negative inputs, identity once x overtakes σ(x).",
   None, False, src=S("Xu & Philip Chen, 2020"))
fn("thlu", "Tanh linear unit", "relu-variants", lambda x: x if x >= 0 else math.tanh(x / 2),
   [("x", "x ≥ 0"), ("tanh(x / 2)", "x < 0")],
   "Identity for positive inputs and a half-rate tanh for negative ones, saturating at −1.",
   None, False, aka="ThLU", src=S("Fu et al., 2022"))
fn("diffelu", "Difference ELU", "relu-variants",
   lambda x: x if x >= 0 else 0.3 * (x * math.exp(x) - 0.1 * math.exp(0.1 * x)),
   [("x", "x ≥ 0"), ("a(x e<sup>x</sup> − b e<sup>bx</sup>)", "x < 0")],
   "An ELU variant whose negative side is a difference of exponentials, giving a dip and a slight negative tail.",
   None, False, aka="DiffELU", src=S("Hu et al., 2020"), note="a = 0.3, b = 0.1 (recommended)", xr=(-8, 3))
fn("polylu", "Polynomial linear unit", "relu-variants", lambda x: x if x >= 0 else 1 / (1 - x) - 1,
   [("x", "x ≥ 0"), ("1 / (1 − x) − 1", "x < 0")],
   "An ELU look-alike whose negative side is rational rather than exponential, saturating at −1. Also published as FPLUS.",
   None, True, aka="PolyLU, FPLUS", src=S("Feng & Yang, 2023"), xr=(-8, 4))
fn("iplu", "Inverse polynomial linear unit", "relu-variants", lambda x: x if x >= 0 else x / (1 + abs(x) ** 2),
   [("x", "x ≥ 0"), ("x / (1 + |x|<sup>a</sup>)", "x < 0")],
   "Identity for positive inputs; negative inputs are damped by 1/(1 + |x|<sup>a</sup>), so their slope stays small.",
   None, True, aka="IpLU", src=S("Alkhouly et al., 2021"), note="a = 2; written as x/(1 + |x|ᵃ) (as printed, the survey's form jumps at 0)", xr=(-6, 4), mono=False)
fn("polu", "Power linear unit", "relu-variants", lambda x: x if x >= 0 else (1 - x) ** -1.5 - 1,
   [("x", "x ≥ 0"), ("(1 − x)<sup>−a</sup> − 1", "x < 0")],
   "An ELU-like unit with a power-law negative side that saturates at −1; a = 1, 1.5 and 2 were tested.",
   None, True, aka="PoLU", src=S("Li et al., 2018"), note="a = 1.5", xr=(-6, 4))
fn("pflu", "Power function linear unit", "relu-variants", lambda x: x * 0.5 * (1 + x / math.sqrt(1 + x * x)),
   "f(x) = x · ½ (1 + x / √(1 + x²))",
   "A Swish-like gated unit whose gate is the algebraic sigmoid ½(1 + x/√(1 + x²)), with no exponentials.",
   None, True, aka="PFLU", src=S("Zhu et al., 2021"))
fn("fpflu", "Faster power function linear unit", "relu-variants",
   lambda x: x if x >= 0 else x + x * x / math.sqrt(1 + x * x),
   [("x", "x ≥ 0"), ("x + x² / √(1 + x²)", "x < 0")],
   "A cheaper PFLU variant: identity for positive inputs and a curve that returns toward −1 for negative ones.",
   None, True, aka="FPFLU", src=S("Zhu et al., 2021"), xr=(-8, 4))
fn("lselu", "Leaky SELU", "relu-variants",
   lambda x: SELU_A * x if x >= 0 else SELU_A * SELU_B * math.expm1(x) + SELU_A * 0.01 * x,
   [("ax", "x ≥ 0"), ("ab(e<sup>x</sup> − 1) + acx", "x < 0")],
   "SELU with an extra leaky term c on the negative side, so it no longer saturates completely.",
   None, False, aka="LSELU", src=S("Chen et al., 2021"), note="a, b as in SELU; c = 0.01")
fn("serlu", "Scaled exponentially-regularised linear unit", "relu-variants",
   lambda x: x if x >= 0 else 2 * x * math.exp(x),
   [("ax", "x ≥ 0"), ("ab x e<sup>x</sup>", "x < 0")],
   "A SELU-style unit whose negative side is x·eˣ, so it bumps down and returns to zero instead of saturating.",
   None, None, aka="SERLU", src=S("Zhang & Li, 2018"), note="a = 1, b = 2 (example values)")
fn("sselu", "Scaled scaled ELU", "relu-variants",
   lambda x: SELU_A * x if x >= 0 else SELU_A * SELU_B * math.expm1(0.5 * x),
   [("ax", "x ≥ 0"), ("ab(e<sup>cx</sup> − 1)", "x < 0")],
   "SELU with the negative inputs rescaled by c before the exponential, which slows its saturation.",
   None, False, aka="sSELU", src=S("Chen et al., 2021"), note="a, b as in SELU; c = 0.5")
fn("rsigelu", "RSigELU", "relu-variants",
   lambda x: x * sig(x) * 0.5 + x if x > 1 else (x if x >= 0 else 0.5 * math.expm1(x)),
   [("x σ(x) a + x", "x > 1"), ("x", "0 ≤ x ≤ 1"), ("a(e<sup>x</sup> − 1)", "x < 0")],
   "ELU-like for negative inputs, identity on [0, 1], and steeper above 1 thanks to an added sigmoid-weighted term. a = 0 gives ReLU.",
   None, False, src=S("Kiliçarslan & Celik, 2021"), note="a = 0.5; middle condition written as 0 ≤ x ≤ 1 (the survey's is a typo)", xr=(-4, 4))
fn("rsigelud", "RSigELUD", "relu-variants",
   lambda x: x * sig(x) * 0.5 + x if x > 1 else (x if x >= 0 else 0.5 * math.expm1(x)),
   [("x σ(x) a + x", "x > 1"), ("x", "0 ≤ x ≤ 1"), ("b(e<sup>x</sup> − 1)", "x < 0")],
   "RSigELU with separate parameters for the positive (a) and negative (b) sides.",
   None, False, src=S("Kiliçarslan & Celik, 2021"), note="a = 0.5, b = 0.5")
fn("hardsrelue", "HardSReLUE", "relu-variants",
   lambda x: 0.5 * x * max(0.0, min(1.0, (x + 1) / 2)) + x if x >= 0 else 0.5 * math.expm1(x),
   [("a x · hardsigmoid(x) + x", "x ≥ 0"), ("a(e<sup>x</sup> − 1)", "x < 0")],
   "ELU on the negative side and an accelerating positive side built from a hard sigmoid, cheap to compute.",
   None, False, src=S("Kiliçarslan, 2022"), note="a = 0.5")
fn("ls-relu", "LS-ReLU", "relu-variants",
   lambda x: x / (1 + abs(x)) if x <= 0 else (x if x <= 1 else math.log(x + 1) + abs(math.log(2) - 1)),
   [("x / (1 + |x|)", "x ≤ 0"), ("x", "0 ≤ x ≤ b"), ("log(ax + 1) + |log(ab + 1) − b|", "x ≥ b")],
   "Softsign for negative inputs, identity up to b, then a logarithmic positive side that compresses large activations.",
   None, False, src=S("Wang et al., 2020"), note="a = 1, b = 1 (natural log)", xr=(-6, 8))

# ---- square-based (survey §1.8)
fn("sqlu", "Square linear unit", "algebraic", lambda x: x if x > 0 else (x + x * x / 4 if x >= -2 else -1.0),
   [("x", "x > 0"), ("x + x²/4", "−2 ≤ x ≤ 0"), ("−1", "x < −2")],
   "An ELU-like unit built from squares instead of exponentials: SQNL's negative half joined to the identity. Designed for low-power hardware.",
   None, True, aka="SQLU", src=S("Wuraola et al., 2021"))
fn("squish", "Squish", "algebraic", lambda x: x + x * x / 32 if x > 0 else (x + x * x / 2 if x >= -2 else 0.0),
   [("x + x²/32", "x > 0"), ("x + x²/2", "−2 ≤ x ≤ 0"), ("0", "x < −2")],
   "A Swish-inspired curve made of quadratic pieces, cheap on hardware without exponentials.",
   None, None, aka="Square swish", src=S("Wuraola et al., 2021"))


# =============================================================================== batch 4
# Remaining fixed functions (survey §1.8–1.40): square-, root-, log- and trig-based activations


def sgn(x):
    return (x > 0) - (x < 0)


fn("sqreu", "Square REU", "algebraic", lambda x: x if x > 0 else (x + x * x / 2 if x >= -2 else 0.0),
   [("x", "x > 0"), ("x + x²/2", "−2 ≤ x ≤ 0"), ("0", "x < −2")],
   "A square-based stand-in for REU: identity for positive inputs and a quadratic dip for negative ones, with no exponentials.",
   None, None, aka="SqREU", src=S("Wuraola et al., 2021"), mono=False)
fn("sqsoftplus", "Square softplus", "algebraic",
   lambda x: x if x > 0.5 else ((x + 0.5) ** 2 / 2 if x >= -0.5 else 0.0),
   [("x", "x > ½"), ("(x + ½)² / 2", "−½ ≤ x ≤ ½"), ("0", "x < −½")],
   "A quadratic-patch replacement for softplus: zero, then a parabola, then the identity, so it is differentiable without exponentials.",
   None, True, aka="SqSoftplus", src=S("Wuraola et al., 2021"),
   note="middle piece written as (x + ½)²/2 so the pieces join (the survey's printed form does not)", xr=(-2, 2))
fn("logsqnl", "Square logistic sigmoid", "algebraic",
   lambda x: 1.0 if x > 2 else (0.5 * (x - x * x / 4) + 0.5 if x >= 0 else (0.5 * (x + x * x / 4) + 0.5 if x >= -2 else 0.0)),
   [("1", "x > 2"), ("½(x − x²/4) + ½", "0 ≤ x ≤ 2"), ("½(x + x²/4) + ½", "−2 ≤ x < 0"), ("0", "x < −2")],
   "SQNL rescaled to (0, 1): a logistic-sigmoid replacement built from two parabolas.",
   "[0, 1]", True, aka="LogSQNL", src=S("Wuraola & Patel, 2018"))
fn("sqmax", "Square softmax", "output", lambda x: (x + 4) ** 2 / ((x + 4) ** 2 + 16),
   "f(z)<sub>j</sub> = (z<sub>j</sub> + c)² / Σ<sub>k</sub> (z<sub>k</sub> + c)²",
   "A softmax replacement that uses squares instead of exponentials, with c = 4. Plotted for two classes with z₂ = 0.",
   None, None, aka="SQMAX", src=S("Wuraola et al., 2021"), note="p₁ for two classes, z₂ = 0, c = 4", xr=(-4, 6))
fn("mef", "Modified Elliott function", "sigmoid-variants", lambda x: x / math.sqrt(1 + x * x) + 0.5,
   "f(x) = x / √(1 + x²) + ½",
   "An algebraic sigmoid (ISRU with a = 1) shifted up by one half.",
   None, True, aka="MEF", src=S("Burhani et al., 2015"), xr=(-8, 8))
fn("sqrt-activation", "Square-root activation", "algebraic", lambda x: sgn(x) * math.sqrt(abs(x)),
   [("√x", "x ≥ 0"), ("−√(−x)", "x < 0")],
   "A signed square root: unbounded but sub-linear, so it compresses large inputs while keeping their sign. Reported to beat tanh and ReLU on CIFAR-10.",
   None, False, aka="SQRT", src=S("Yang et al., 2018"))
fn("ssaf", "Scaled square-root activation", "algebraic", lambda x: sgn(x) * math.sqrt(2 * 2 * abs(x)),
   [("√(2ax)", "x ≥ 0"), ("−√(−2ax)", "x < 0")],
   "The signed square root with a scale parameter a.",
   None, False, aka="SSAF", src=S("Khachumov et al., 2023"), note="a = 2")
fn("mishra", "Mishra activation", "algebraic",
   lambda x: 0.5 * (x / (1 + abs(x))) ** 2 + 0.5 * x / (1 + abs(x)),
   "f(x) = ½ s(x)² + ½ s(x),  s(x) = x / (1 + |x|)",
   "A quadratic polynomial of the softsign: bounded, non-monotonic and asymmetric, approaching 1 for large positive inputs and 0 for large negative ones.",
   None, True, src=S("Mishra et al., 2021"), xr=(-10, 10))
fn("sbaf", "Saha-Bora activation", "algebraic", lambda x: 1 / (1 + 0.98 * x ** 0.5 * (1 - x) ** 0.5) if 0 <= x <= 1 else None,
   "f(x) = 1 / (1 + k x<sup>α</sup>(1 − x)<sup>1 − α</sup>)",
   "Proposed for classifying the habitability of exoplanets; defined on [0, 1], where it dips in the middle and equals 1 at both ends.",
   None, True, aka="SBAF", src=S("Saha et al., 2018"), note="k = 0.98, α = 0.5; defined for 0 ≤ x ≤ 1", xr=(0, 1), mono=False)
fn("symlog", "Logarithmic activation (symlog)", "algebraic", lambda x: sgn(x) * math.log1p(abs(x)),
   [("ln(x + 1)", "x ≥ 0"), ("−ln(−x + 1)", "x < 0")],
   "A signed logarithm that compresses large values in both directions. Proposed early as the LAF and later as symlog, which DreamerV3 uses to squash rewards and values.",
   None, True, aka="LAF, symlog", src=S("Bilski, 2000; Hafner et al., 2023"), xr=(-10, 10))
fn("symexp", "Symexp", "algebraic", lambda x: sgn(x) * math.expm1(abs(x)),
   "f(x) = sgn(x) (e<sup>|x|</sup> − 1)",
   "The inverse of symlog: expands values symmetrically. DreamerV3 uses it to decode predictions made in symlog space.",
   None, True, src=S("Hafner et al., 2023"), xr=(-3, 3))


def _spocu_r(v):
    return v ** 3 * (v ** 5 - 2 * v ** 4 + 2)


def _spocu_h(v, d=1.0):
    return _spocu_r(d) if v >= d else (_spocu_r(v) if v >= 0 else v)


fn("spocu", "SPOCU", "algebraic", lambda x: _spocu_h(x / 1.0 + 0.5) - _spocu_h(0.5),
   "f(x) = a h(x/c + b) − a h(b),  h from r(x) = x³(x⁵ − 2x⁴ + 2)",
   "Scaled polynomial constant unit: linear for negative inputs, a high-order polynomial in the middle and constant beyond d. Designed for self-normalising networks.",
   None, None, src=S("Kiseľák et al., 2020"), note="a = 1, b = 0.5, c = 1, d = 1 (example values)", xr=(-2, 2), mono=False)
fn("puaf", "Polynomial universal activation", "algebraic",
   lambda x: x if x > 2 else (x * (2 + x) / ((2 + x) + (2 - x)) if x >= -2 else 0.0),
   [("x<sup>a</sup>", "x > c"), ("x<sup>a</sup> (c + x)<sup>b</sup> / ((c + x)<sup>b</sup> + (c − x)<sup>b</sup>)", "|x| ≤ c"), ("0", "x < −c")],
   "A polynomial family that can approximate the sigmoid, ReLU and Swish by choosing a, b and c (a = 1, b = 0, c = 0 gives ReLU).",
   None, None, aka="PUAF", src=S("Hwang & Kim, 2023"), note="a = 1, b = 1, c = 2", mono=False)
fn("psoftplus", "Parametric softplus", "relu-variants", lambda x: softplus(x) - math.log(2),
   "f(x) = a (ln(e<sup>x</sup> + 1) − b)",
   "Softplus with a scale a and a shift b; the shift lets the mean output move toward zero.",
   None, True, aka="PSoftplus", src=S("Sun et al., 2019"), note="a = 1, b = ln 2")
fn("softplusplus", "Soft++", "relu-variants", lambda x: softplus(x) + x / 2 - math.log(2),
   "f(x) = ln(1 + e<sup>ax</sup>) + x/b − ln 2",
   "A non-saturating softplus extension: an added linear term keeps negative inputs from flattening out. a = 1, b = 2 worked well.",
   None, True, aka="Soft++", src=S("Ciuparu et al., 2020"), note="a = 1, b = 2")
fn("aranda-ordaz", "Aranda-Ordaz", "sigmoid-variants", lambda x: 1 - (1 + 2 * math.exp(x)) ** (-0.5) if x < 700 else 1.0,
   "f(x) = 1 − (1 + a e<sup>x</sup>)<sup>−1/a</sup>",
   "An asymmetric link function from statistics that includes the logistic (a = 1) and complementary log-log (a → 0) curves as special cases.",
   None, True, src=S("Gomes & Ludermir, 2013; Aranda-Ordaz, 1981"), note="a = 2")
fn("bfire", "Bi-firing", "relu-variants",
   lambda x: x - 0.5 if x > 1 else (-x - 0.5 if x < -1 else x * x / 2),
   [("x − a/2", "x > a"), ("x² / 2a", "−a ≤ x ≤ a"), ("−x − a/2", "x < −a")],
   "A smoothed absolute value: a parabola near zero joined to straight lines, so it fires for both large positive and large negative inputs.",
   None, True, aka="bfire", src=S("Li et al., 2013"), note="a = 1; middle condition written as −a ≤ x ≤ a (the survey's is a typo)")
fn("bbfire", "Bounded bi-firing", "relu-variants",
   lambda x: 3.0 if x < -3.5 else (-x - 0.5 if x < -1 else (x * x / 2 if x <= 1 else (x - 0.5 if x <= 3.5 else 3.0))),
   [("b", "x < −b − a/2"), ("−x − a/2", "−b − a/2 ≤ x < −a"), ("x² / 2a", "−a ≤ x ≤ a"), ("x − a/2", "a < x ≤ b + a/2"), ("b", "x > b + a/2")],
   "Bi-firing capped at b on both sides, giving an inverted bell shape.",
   None, False, aka="bbfire", src=S("Liew et al., 2016"), note="a = 1, b = 3; conditions written consistently (the survey's contain typos)", xr=(-6, 6))
fn("pmaf", "Piecewise Mexican-hat", "periodic",
   lambda x: (2 / math.sqrt(3) * math.pi ** -0.25) * (1 - (x + 4) ** 2) * math.exp(-(x + 4) ** 2 / 2) if x < 0
   else (2 / math.sqrt(3) * math.pi ** -0.25) * (1 - (x - 4) ** 2) * math.exp(-(x - 4) ** 2 / 2),
   [("C (1 − (x + a)²) e<sup>−(x + a)²/2</sup>", "x < 0"), ("C (1 − (x − a)²) e<sup>−(x − a)²/2</sup>", "x ≥ 0")],
   "Two Mexican-hat wavelets centred at ±a (C = 2π<sup>−1/4</sup>/√3), so the unit responds to inputs near a or −a.",
   None, None, aka="PMAF", src=S("Liu et al., 2016"), note="a = 4", xr=(-8, 8), mono=False)
fn("prbf", "Piecewise radial basis function", "periodic",
   lambda x: math.exp(-(x - 6) ** 2) if x >= 3 else (math.exp(-x * x) if x > -3 else math.exp(-(x + 6) ** 2)),
   [("exp(−(x − 2a)² / b²)", "x ≥ a"), ("exp(−x² / b²)", "−a < x < a"), ("exp(−(x + 2a)² / b²)", "x ≤ −a")],
   "Three Gaussian bumps centred at −2a, 0 and 2a, so the unit responds to several separate input ranges.",
   None, None, aka="PRBF", src=S("Liu et al., 2016"), note="a = 3, b = 1", xr=(-9, 9), mono=False)
fn("comb-h-sine", "Comb-H-sine", "algebraic", lambda x: math.sinh(0.5 * x) + math.asinh(0.5 * x),
   "f(x) = sinh(ax) + sinh<sup>−1</sup>(ax)",
   "Found by an evolutionary search: steep near zero, flattening a little, then growing exponentially. Its finders report it beating ReLU and tanh in LSTMs.",
   None, True, src=S("Vijayaprabakaran & Sathiyamurthy, 2022"), note="a = 0.5", xr=(-6, 6))
fn("m-arcsinh", "Modified arcsinh", "algebraic", lambda x: math.asinh(x) * math.sqrt(abs(x)) / 12,
   "f(x) = (1/12) sinh<sup>−1</sup>(x) √|x|",
   "A scaled product of arcsinh and the square root of |x|; usable both as a network activation and as a support-vector-machine kernel.",
   None, None, aka="m-arcsinh", src=S("Parisi, 2020"), xr=(-10, 10))
fn("hyper-sinh", "Hyper-sinh", "algebraic", lambda x: math.sinh(x) / 3 if x > 0 else x ** 3 / 4,
   [("sinh(x) / 3", "x > 0"), ("x³ / 4", "x ≤ 0")],
   "A scaled hyperbolic sine for positive inputs and a cubic for negative ones.",
   None, None, src=S("Parisi et al., 2021"), xr=(-3, 3))
fn("arctid", "Arctid", "algebraic", lambda x: math.atan(x) ** 2 - x,
   "f(x) = arctan(x)² − x",
   "Squared arctan minus the identity: a mostly decreasing curve with a bump near zero.",
   None, True, src=S("Xu & Philip Chen, 2020"))
fn("shifted-sine", "Scaled and shifted sine", "periodic", lambda x: 0.5 * math.sin(1.2 * x) + 0.3,
   "f(x) = 0.5 sin(ax) + 0.3",
   "A sine with reduced amplitude and an upward offset; frequencies a from 0.2 to 4 were compared.",
   None, True, src=S("Gashler & Ashmore, 2014"), note="a = 1.2", xr=(-6.3, 6.3))
fn("one-minus-cosine", "1 − cosine", "periodic", lambda x: 1 - math.cos(x),
   "f(x) = 1 − cos(x)",
   "A non-negative cosine bump that is zero at the origin.",
   "[0, 2]", True, src=S("Daskin, 2018"), xr=(-6.3, 6.3))
fn("cosid", "Cosid", "periodic", lambda x: math.cos(x) - x,
   "f(x) = cos(x) − x",
   "A decreasing line with a cosine ripple on top.",
   None, True, src=S("Xu & Philip Chen, 2020"), xr=(-6.3, 6.3))
fn("sinp", "SinP", "periodic", lambda x: math.sin(x) - x,
   "f(x) = sin(x) − ax",
   "A sine ripple on a decreasing line; slopes a = 1, 1.5 and 2 were used.",
   None, True, src=S("Chan et al., 2018"), note="a = 1", xr=(-6.3, 6.3))
fn("asu", "Amplifying sine unit", "periodic", lambda x: x * math.sin(x),
   "f(x) = x · sin(x)",
   "The sine counterpart of the growing cosine unit: an even function oscillating with growing amplitude.",
   None, True, aka="ASU", src=S("Rahman et al., 2023"), xr=(-6.3, 6.3))
fn("ssu", "Shifted sinc unit", "periodic", lambda x: math.pi * sinc(x - math.pi),
   "f(x) = π sinc(x − π)",
   "A sinc shifted right by π and scaled by π, so its main lobe sits at x = π.",
   None, True, aka="SSU", src=S("Gustineli, 2022"), note="sinc(x) = sin(x)/x", xr=(-8, 12))
fn("dsu", "Decaying sine unit", "periodic", lambda x: math.pi / 2 * (sinc(x - math.pi) - sinc(x + math.pi)),
   "f(x) = (π/2)(sinc(x − π) − sinc(x + π))",
   "The difference of two shifted sincs: an odd, decaying oscillation.",
   None, True, aka="DSU", src=S("Gustineli, 2022"), note="sinc(x) = sin(x)/x", xr=(-12, 12))
fn("hclsh", "HcLSH", "algebraic",
   lambda x: math.log(math.cosh(x) + x * math.cosh(x / 2)) if x >= 0 else math.log(math.cosh(x)) + x,
   [("ln(cosh(x) + x cosh(x/2))", "x ≥ 0"), ("ln(cosh(x)) + x", "x < 0")],
   "Hyperbolic cosine linearised squashing function: log-cosh based, close to ReLU but smooth, with a slight negative lobe.",
   None, None, src=S("Abdel-Nabi et al., 2023"), xr=(-4, 4))
fn("polyexp", "Polyexp", "algebraic", lambda x: (x * x + x) * math.exp(-0.5 * x * x),
   "f(x) = (ax² + bx + c) e<sup>−dx²</sup>",
   "A quadratic multiplied by a Gaussian envelope, so it responds only near zero.",
   None, True, src=S("Efe, 2008"), note="a = 1, b = 1, c = 0, d = 0.5 (example values)", mono=False)
fn("negative-exponential", "Negative exponential", "algebraic", lambda x: math.exp(-x),
   "f(x) = e<sup>−x</sup>",
   "A decreasing exponential used as an activation in a physics-oriented network.",
   None, True, src=S("Pan et al., 2020"), xr=(-2.5, 4))
fn("e-tanh", "E-Tanh", "algebraic", lambda x: 0.5 * math.exp(x) * math.tanh(x),
   "f(x) = a e<sup>x</sup> tanh(x)",
   "An exponentially weighted tanh: a small negative dip for negative inputs and explosive growth for positive ones.",
   None, True, src=S("Kalaiselvi et al., 2022"), note="a = 0.5", xr=(-4, 2.5))
fn("evolved-tanh-relu", "Evolved tanh–ReLU", "algebraic", lambda x: math.tanh(x * x) + relu(x),
   "f(x) = a (tanh(x²) + ReLU(x))",
   "Found by neuroevolution as the best recurrent activation for LSTMs on a human-activity dataset: ReLU plus a saturating bump.",
   None, None, src=S("Vijayaprabakaran & Sathiyamurthy, 2021"), note="a = 1", mono=False)
fn("wave", "Wave", "periodic", lambda x: (1 - x * x) * math.exp(-x * x),
   "f(x) = (1 − x²) e<sup>−ax²</sup>",
   "A Mexican-hat shape with a single width parameter; a = ½ gives the Ricker wavelet.",
   None, True, src=S("Efe, 2008"), note="a = 1", xr=(-3, 3))
fn("ncu", "Non-monotonic cubic unit", "algebraic", lambda x: x - x ** 3,
   "f(x) = x − x³",
   "A cubic that rises near zero and falls beyond ±1/√3, making it strongly non-monotonic.",
   None, True, aka="NCU", src=S("Gustineli, 2022"), xr=(-1.6, 1.6))
fn("squ", "Shifted quadratic unit", "algebraic", lambda x: x * x + x,
   "f(x) = x² + x",
   "A parabola with its minimum at x = −½; simple and non-monotonic.",
   None, True, aka="SQU", src=S("Gustineli, 2022"), xr=(-3, 2))


# =============================================================================== batch 5
# Adaptive activation functions, part 1 (survey §2.1–2.2: TAAF and the ReLU-based adaptive family).
# Trainable parameters are drawn at their published initial values where given, otherwise at example values.

def elu_(x, a=1.0):
    return x if x >= 0 else a * math.expm1(x)


fn("taaf", "Transformative adaptive activation function", "adaptive-frameworks",
   lambda x: 1.5 * math.tanh(2 * x + 0.5) + 0.2,
   "g(f, x) = α f(βx + γ) + δ",
   "Wraps any activation f in four trainable parameters that scale and shift it horizontally and vertically, which covers many published adaptive functions as special cases. Plotted for f = tanh.",
   None, True, aka="TAAF", src=S("Kunc et al., 2021"), note="f = tanh, α = 1.5, β = 2, γ = 0.5, δ = 0.2 (example values)")
fn("prelu-plus", "Positive PReLU", "adaptive-rectifiers", lambda x: 1.5 * x if x >= 0 else 0.0,
   [("ax", "x ≥ 0"), ("0", "x < 0")],
   "ReLU with a trainable positive slope a: the adaptive version of the sloped ReLU.",
   None, False, aka="PReLU⁺", src=S("Dai et al., 2022"), note="a = 1.5 (example)")
fn("margin-relu", "Margin ReLU", "adaptive-rectifiers", lambda x: max(x, -0.5),
   "f(x) = max(x, a)",
   "A shifted ReLU whose floor a is set per channel from the expected value of the negative responses. Used in knowledge distillation.",
   None, False, aka="MarReLU", src=S("Heo et al., 2019"), note="a = −0.5 (example)")
fn("rprelu", "React-PReLU", "adaptive-rectifiers", lambda x: x - 0.2 - 0.1 if x >= 0.2 else 0.25 * (x - 0.2) - 0.1,
   [("x − a + b", "x ≥ a"), ("c(x − a) + b", "x < a")],
   "PReLU with learned horizontal (a) and vertical (b) shifts per channel. Introduced in ReActNet for binary neural networks.",
   None, False, aka="RPReLU", src=S("Liu et al., 2020"), note="a = 0.2, b = −0.1, c = 0.25 (example)")
fn("sau", "Smooth activation unit", "adaptive-rectifiers",
   lambda x: 1 / (2 * 2) * math.sqrt(2 / math.pi) * math.exp(-(2 * x) ** 2 / 2) + (1 + 1 / 4) / 2 * x + (1 - 1 / 4) / 2 * x * math.erf(2 * x / math.sqrt(2)),
   "f(x) = (PReLU<sub>a</sub> ∗ φ<sub>b</sub>)(x)",
   "PReLU smoothed by convolving it with a Gaussian of learned width, which gives a closed form in terms of erf and exp.",
   None, True, aka="SAU", src=S("Biswas et al., 2022"), note="negative slope 1/a = 0.25, b = 2 (example)")
fn("smu", "Smooth maximum unit", "adaptive-rectifiers",
   lambda x: ((1 + 0.25) * x + (1 - 0.25) * x * math.erf(1 * (1 - 0.25) * x)) / 2,
   "f(x) = [(1 + a)x + (1 − a)x · erf(b(1 − a)x)] / 2",
   "A smooth leaky ReLU built from an erf approximation of |x|, i.e. a smooth max(x, ax). The trainable b controls how sharp the corner is.",
   None, True, aka="SMU", src=S("Biswas et al., 2022"), note="a = 0.25, b = 1 (example)")
fn("lelelu", "Leaky learnable ReLU", "adaptive-rectifiers", lambda x: 1.5 * x if x >= 0 else 0.015 * x,
   [("ax", "x ≥ 0"), ("0.01ax", "x < 0")],
   "Leaky ReLU multiplied by a trainable scale a, so both slopes are learned together.",
   None, False, aka="LeLeLU", src=S("Maniatopoulos & Mitianoudis, 2021"), note="a = 1.5 (example)")
fn("preu", "Parametric rectified exponential unit", "adaptive-rectifiers",
   lambda x: 1.2 * x if x >= 0 else 1.2 * x * math.exp(0.5 * x),
   [("ax", "x ≥ 0"), ("ax e<sup>bx</sup>", "x < 0")],
   "REU with trainable scale a and decay b, keeping a Swish-like dip for negative inputs.",
   None, None, aka="PREU", src=S("Ying et al., 2019"), note="a = 1.2, b = 0.5 (example)", xr=(-8, 4))
fn("frelu-flexible", "Flexible ReLU", "adaptive-rectifiers", lambda x: relu(x) - 0.4,
   "f(x) = ReLU(x) + b",
   "ReLU with a learned vertical shift b, which lets outputs go negative and pushes the mean toward zero. (The horizontal shift is absorbed by the preceding bias.)",
   None, False, aka="FReLU", src=S("Qiu et al., 2018"), note="b = −0.4 (example)")
fn("shilu", "Adaptive shifted ReLU", "adaptive-rectifiers", lambda x: 1.2 * relu(x) - 0.2,
   "f(x) = a · max(0, x) + b",
   "ReLU with a trainable slope a and vertical shift b per neuron.",
   None, False, aka="ShiLU", src=S("Pishchik, 2023"), note="a = 1.2, b = −0.2 (example)")
fn("adaptive-hardtanh", "Adaptive HardTanh", "adaptive-rectifiers", lambda x: max(-1.0, min(1.0, 2 * (x - 0.5))),
   "f(x) = HardTanh(a<sub>t</sub>(x − b))",
   "HardTanh whose slope a<sub>t</sub> grows on a schedule over training epochs while the offset b is learned, gradually turning it into a sign function for binary networks.",
   None, False, src=S("Liu et al., 2021"), note="a_t = 2, b = 0.5 (example)")
fn("arelu", "Attention-based ReLU", "adaptive-rectifiers", lambda x: (1 + sig(0)) * x if x >= 0 else 0.25 * x,
   [("(1 + σ(b))x", "x ≥ 0"), ("C(a)x", "x < 0")],
   "ReLU whose positive slope is amplified by a learned sigmoid gate and whose negative slope is a learned value clipped to [0.01, 0.99]. One pair of parameters per layer.",
   None, False, aka="AReLU", src=S("Chen et al., 2020"), note="a = 0.25, b = 0 (example)")
fn("dprelu", "Dual parametric ReLU", "adaptive-rectifiers", lambda x: 1.2 * x if x >= 0 else 0.2 * x,
   [("ax", "x ≥ 0"), ("bx", "x < 0")],
   "PReLU with trainable slopes on both sides (initialised to 1 and 0.01). Also published as 'fully parametric ReLU'.",
   None, False, aka="DPReLU", src=S("Balaji et al., 2020"), note="a = 1.2, b = 0.2 (example)")
fn("dual-line", "Dual Line", "adaptive-rectifiers", lambda x: x - 0.22 if x >= 0 else 0.01 * x - 0.22,
   [("ax + m", "x ≥ 0"), ("bx + m", "x < 0")],
   "DPReLU plus a trainable shift m (initialised to −0.22).",
   None, False, src=S("Balaji et al., 2020"), note="a = 1, b = 0.01, m = −0.22 (initial values)")
fn("pilu", "Piecewise linear unit", "adaptive-rectifiers",
   lambda x: 1.2 * x + 0.5 * (1 - 1.2) if x >= 0.5 else 0.2 * x + 0.5 * (1 - 0.2),
   [("ax + c(1 − a)", "x ≥ c"), ("bx + c(1 − b)", "x < c")],
   "Two lines with trainable slopes meeting at a trainable point (c, c). Generalises ReLU, leaky ReLU, PReLU, DPReLU and Dual Line.",
   None, False, aka="PiLU", src=S("Inturrisi et al., 2021"), note="a = 1.2, b = 0.2, c = 0.5 (example)")
fn("paired-relu", "Paired ReLU", "adaptive-rectifiers",
   [("max(ax − b, 0)", lambda x: max(0.5 * x, 0.0)), ("max(cx − d, 0)", lambda x: max(-0.5 * x, 0.0))],
   "f(x) = [max(ax − b, 0), max(cx − d, 0)]",
   "Like CReLU it outputs two channels, but each has a trainable scale and threshold (scales initialised to 0.5 and −0.5).",
   "[0, ∞)²", False, src=S("Tang et al., 2018"), note="a = 0.5, c = −0.5, b = d = 0 (initial values)", mono=False)
fn("tent", "Tent", "adaptive-rectifiers", lambda x: max(0.0, 1 - abs(x)),
   "f(x) = max(0, a − |x|)",
   "A triangular bump of trainable width a, which responds only to inputs near zero. Proposed to improve robustness to adversarial and unknown inputs.",
   None, False, src=S("Rozsa & Boult, 2019"), note="a = 1 (recommended initial value)", xr=(-3, 3))
fn("hat", "Hat", "adaptive-rectifiers", lambda x: 0.0 if x < 0 else (x if x <= 1 else (2 - x if x <= 2 else 0.0)),
   [("0", "x < 0"), ("x", "0 ≤ x ≤ a/2"), ("a − x", "a/2 ≤ x ≤ a"), ("0", "x > a")],
   "A triangle on the positive side only: it rises to a/2 and falls back to zero at a.",
   None, False, src=S("Wang et al., 2022"), note="a = 2; falling piece written as a − x (the survey repeats x)", xr=(-2, 4))
fn("rmaf", "ReLU memristor-like activation", "adaptive-rectifiers",
   lambda x: (1 / (0.25 * (1 + math.exp(-x)) + 0.75) ** 1) * x if x > -700 else 0.0,
   "f(x) = [b / (0.25(1 + e<sup>−x</sup>) + 0.75)<sup>c</sup>] a x",
   "A Swish-like unit modelled on memristor behaviour that keeps some negative information.",
   None, None, aka="RMAF", src=S("Yu et al., 2020"), note="a = 1, b = 1, c = 1 (example)")
fn("ptelu", "Parametric tanh linear unit", "adaptive-rectifiers", lambda x: x if x >= 0 else 0.5 * math.tanh(2 * x),
   [("x", "x ≥ 0"), ("a tanh(bx)", "x < 0")],
   "Identity for positive inputs and a trainable scaled tanh for negative ones, saturating at −a.",
   None, None, aka="PTELU", src=S("Gupta & Duggal, 2017"), note="a = 0.5, b = 2 (example)")
fn("talu", "Tangent linear unit", "adaptive-rectifiers",
   lambda x: x if x >= 0 else (math.tanh(x) if x > -1.5 else math.tanh(-1.5)),
   [("x", "x ≥ 0"), ("tanh(x)", "a < x < 0"), ("tanh(a)", "x ≤ a")],
   "Identity for positive inputs, tanh down to a threshold a < 0, then constant.",
   None, False, aka="TaLU", src=S("Mercioni & Holban, 2021"), note="a = −1.5")
fn("ptalu", "PTaLU", "adaptive-rectifiers",
   lambda x: x if x >= 1 else (math.tanh(x) if x > -0.75 else math.tanh(-0.75)),
   [("x", "x ≥ b"), ("tanh(x)", "a < x < b"), ("tanh(a)", "x ≤ a")],
   "TaLU with a second trainable threshold b above which the function becomes the identity. As written it jumps at x = b.",
   None, False, src=S("Mercioni & Holban, 2021"), note="a = −0.75, b = 1 (initial values)")
fn("tanhlu", "TanhLU", "adaptive-rectifiers", lambda x: math.tanh(x) + 0.5 * x,
   "f(x) = a tanh(bx) + cx",
   "A trainable blend of tanh and a linear term: tanh-shaped near zero, linear far away.",
   None, True, src=S("Shen et al., 2022"), note="a = 1, b = 1, c = 0.5 (example)", xr=(-6, 6))
fn("telu", "TeLU", "adaptive-rectifiers", lambda x: x * math.tanh(elu_(x)),
   "f(x) = x · tanh(ELU(ax))",
   "A Mish/TanhExp relative that gates x with tanh of an ELU, with a trainable input scale a.",
   None, True, src=S("Mercioni & Holban, 2020"), note="a = 1")
fn("trelu-tanh", "Tanh-based ReLU", "adaptive-rectifiers", lambda x: x if x >= 0 else math.tanh(2 * x),
   [("x", "x ≥ 0"), ("tanh(bx)", "x < 0")],
   "A special case of PTELU with a = 1: identity for positive inputs and a trainable-slope tanh for negative ones.",
   None, None, aka="TReLU", src=S("Zhang et al., 2019"), note="b = 2 (example)")


def _reltanh(x, a=-1.5, b=0.5):
    d = lambda v: 4 / (math.exp(v) + math.exp(-v)) ** 2
    if x <= a:
        return d(a) * (x - a) + math.tanh(a)
    if x >= b:
        return d(b) * (x - b) + math.tanh(b)
    return math.tanh(x)


fn("reltanh", "Rectified linear tanh", "adaptive-rectifiers", _reltanh,
   [("tanh′(a)(x − a) + tanh(a)", "x ≤ a"), ("tanh(x)", "a < x < b"), ("tanh′(b)(x − b) + tanh(b)", "x ≥ b")],
   "Tanh between two trainable thresholds, continued outside them by straight lines tangent to the tanh, so the gradient never vanishes.",
   None, True, aka="ReLTanh", src=S("Wang et al., 2019"), note="a = −1.5, b = 0.5 (example)", xr=(-6, 6))
fn("blu", "Bendable linear unit", "adaptive-rectifiers",
   [("a = −0.5", lambda x: -0.5 * (math.sqrt(x * x + 1) - 1) + x), ("a = 0.5", lambda x: 0.5 * (math.sqrt(x * x + 1) - 1) + x),
    ("a = 1", lambda x: (math.sqrt(x * x + 1) - 1) + x)],
   "f(x) = a(√(x² + 1) − 1) + x",
   "Interpolates between the identity (a = 0) and a smooth rectifier (a = 1) with a trainable a ∈ [−1, 1]. Being able to become the identity lets a layer effectively be skipped.",
   None, True, aka="BLU", src=S("Godfrey, 2019"), mono=False)
fn("reblu", "Rectified BLU", "adaptive-rectifiers", lambda x: 0.5 * (math.sqrt(x * x + 1) - 1) + x if x > 0 else 0.0,
   [("a(√(x² + 1) − 1) + x", "x > 0"), ("0", "x ≤ 0")],
   "BLU on the positive side and zero on the negative side.",
   None, False, aka="ReBLU", src=S("Dai et al., 2022"), note="a = 0.5")
fn("delu", "DELU", "adaptive-rectifiers",
   lambda x: (0 + 0.5) * x + abs(math.exp(-x) - 1) if x >= 0 else x * sig(x),
   [("(a + 0.5)x + |e<sup>−x</sup> − 1|", "x ≥ 0"), ("x σ(x)", "x < 0")],
   "SiLU for negative inputs and a trainable-slope line plus a saturating term for positive ones.",
   None, None, src=S("Pishchik, 2023"), note="a = 0")
fn("sc-mish", "Soft clipping Mish", "adaptive-rectifiers", lambda x: max(0.0, x * math.tanh(softplus(x))),
   "f(x) = max(0, x · tanh(softplus(ax)))",
   "Mish with its negative lobe clipped to zero. With a trainable a it is called SCL-mish (initialised to a = 0.25).",
   None, False, aka="SC-mish, SCL-mish", src=S("Mercioni & Holban, 2021"), note="a = 1")
fn("sc-swish", "Soft clipping Swish", "adaptive-rectifiers", lambda x: max(0.0, x * sig(x)),
   "f(x) = max(0, x · σ(x))",
   "Swish with its negative lobe clipped to zero.",
   None, False, aka="SC-swish", src=S("Mercioni & Holban, 2022"))
fn("p-swish", "Parametric swish (p-swish)", "adaptive-rectifiers",
   lambda x: 1 * x * sig(1 * x) if x <= 0 else x,
   [("a x σ(bx)", "x ≤ c"), ("x", "x > c")],
   "Swish below a trainable threshold c and the identity above it. Also proposed as R_S with a = 1.",
   None, None, src=S("Mercioni & Holban, 2020"), note="a = 1, b = 1, c = 0 (initial values)")
fn("edelu", "Extended ELU", "adaptive-rectifiers",
   lambda x: x if x >= 0.5 else math.expm1(x) / ((math.exp(0.5) - 1) / 0.5),
   [("x", "x ≥ c"), ("(e<sup>ax</sup> − 1) / b", "x < c")],
   "An ELU whose switch point c is trainable, with b tied to a and c (bc = e<sup>ac</sup> − 1) so the pieces meet.",
   None, None, aka="EDELU", src=S("Çatalbaş & Morgül, 2023"), note="a = 1, c = 0.5, b = (e^0.5 − 1)/0.5")
fn("prelu-pelu-mix", "Gated PReLU–PELU mix", "adaptive-rectifiers",
   lambda x: sig(x) * (x if x >= 0 else 0.25 * x) + (1 - sig(x)) * (x if x >= 0 else math.expm1(x)),
   "f(x) = σ(ax) PReLU(x) + (1 − σ(ax)) PELU(x)",
   "A learned, input-dependent blend of PReLU and PELU, one of several activation-mixing schemes compared in the same work.",
   None, None, src=S("Qian et al., 2018"), note="a = 1, PReLU slope 0.25, PELU a = b = 1")
fn("felu", "Fast ELU", "adaptive-rectifiers", lambda x: x if x >= 0 else 1.0 * (2 ** (x / math.log(2)) - 1),
   [("x", "x ≥ 0"), ("a(2<sup>x / ln 2</sup> − 1)", "x < 0")],
   "ELU rewritten with a power of two, which allows a fast bit-level approximation of the exponential. Mathematically equal to ELU.",
   None, True, aka="FELU", src=S("Qiumei et al., 2019"), note="a = 1")
fn("p-plus-felu", "P+FELU", "adaptive-rectifiers", lambda x: x - 0.2 if x >= 0 else (2 ** (x / math.log(2)) - 1) - 0.2,
   [("x + b", "x ≥ 0"), ("a(2<sup>x / ln 2</sup> − 1) + b", "x < 0")],
   "FELU with an added trainable vertical shift b.",
   None, True, src=S("Adem, 2022"), note="a = 1, b = −0.2 (example)")
fn("mpelu", "Multiple parametric ELU", "adaptive-rectifiers", lambda x: x if x >= 0 else math.expm1(2 * x),
   [("x", "x ≥ 0"), ("a(e<sup>bx</sup> − 1)", "x < 0")],
   "ELU with trainable saturation a and steepness b; ReLU, ELU and some PELU settings are special cases.",
   None, False, aka="MPELU", src=S("Li et al., 2018"), note="a = 1, b = 2 (example)")
fn("p-e2-relu", "P-E2-ReLU", "adaptive-rectifiers",
   lambda x: 0.4 * relu(x) + 0.3 * elu_(x) + (1 - 0.4 - 0.3) * (-elu_(-x)),
   "f(x) = a ReLU(x) + b ELU(x) + (1 − a − b)(−ELU(−x))",
   "A trainable mix of ReLU, ELU and a mirrored ELU, giving a flexible piecewise-smooth shape.",
   None, None, src=S("Jie et al., 2021"), note="a = 0.4, b = 0.3 (initial values)")
fn("erfrelu", "Erf-based ReLU", "adaptive-rectifiers", lambda x: x if x >= 0 else 0.5 * math.erf(x),
   [("x", "x ≥ 0"), ("a erf(x)", "x < 0")],
   "An ELU-like unit whose negative side is a trainable multiple of erf(x), saturating at −a.",
   None, None, aka="ErfReLU", src=S("Rajanand & Singh, 2023"), note="a = 0.5 (example)")
fn("pselu", "Parametric SELU", "adaptive-rectifiers",
   lambda x: SELU_A * x if x >= 0 else SELU_A * SELU_B * math.expm1(x),
   [("ax", "x ≥ 0"), ("ab(e<sup>x</sup> − 1)", "x < 0")],
   "SELU with its scale and saturation constants made trainable per neuron.",
   None, False, aka="PSELU", src=S("Pratama & Kang, 2020"), note="a, b initialised to the SELU constants")
fn("lpselu", "Leaky parametric SELU", "adaptive-rectifiers",
   lambda x: SELU_A * x if x >= 0 else SELU_A * SELU_B * math.expm1(x) + 0.01 * x,
   [("ax", "x ≥ 0"), ("ab(e<sup>x</sup> − 1) + cx", "x < 0")],
   "PSELU with a small leaky term c so negative inputs keep a gradient.",
   None, False, aka="LPSELU", src=S("Pratama & Kang, 2020"), note="a, b = SELU constants, c = 0.01")
fn("lpselu-rp", "LPSELU with reposition", "adaptive-rectifiers",
   lambda x: SELU_A * x - 0.2 if x >= 0 else SELU_A * SELU_B * math.expm1(x) + 0.01 * x - 0.2,
   [("ax + m", "x ≥ 0"), ("ab(e<sup>x</sup> − 1) + cx + m", "x < 0")],
   "LPSELU plus a trainable vertical shift m.",
   None, False, aka="LPSELU_RP", src=S("Pratama & Kang, 2020"), note="a, b = SELU constants, c = 0.01, m = −0.2 (example)")


# =============================================================================== batch 6
# Adaptive activation functions, part 2 (survey §2.2 end, §2.3 sigmoid-based, §2.4 adaptive SiLU-type,
# and the standalone adaptive functions up to the EIS family)

fn("shelu", "Horizontally shifted ELU", "relu-variants", lambda x: x + 0.5 if x + 0.5 >= 0 else math.expm1(x + 0.5),
   [("x + b", "x + b ≥ 0"), ("a(e<sup>x + b</sup> − 1)", "x + b < 0")],
   "ELU moved left by a fixed shift b.",
   None, True, aka="ShELU", src=S("Grelsson & Felsberg, 2018"), note="a = 1, b = 0.5")
fn("svelu", "Vertically shifted ELU", "relu-variants", lambda x: x - 0.3 if x >= 0 else math.expm1(x) - 0.3,
   [("x + b", "x ≥ 0"), ("a(e<sup>x</sup> − 1) + b", "x < 0")],
   "ELU moved down by a fixed shift b.",
   None, True, aka="SvELU", src=S("Grelsson & Felsberg, 2018"), note="a = 1, b = −0.3")
fn("pshelu", "Parametric shifted ELU (horizontal)", "adaptive-rectifiers",
   lambda x: 1.5 / 1 * (x + 0.5) if x + 0.5 >= 0 else 1.5 * math.expm1((x + 0.5) / 1),
   [("(a/b)(x + c)", "x + c ≥ 0"), ("a(e<sup>(x + c)/b</sup> − 1)", "x + c < 0")],
   "PELU with a trainable horizontal shift c.",
   None, True, aka="PShELU", src=S("Grelsson & Felsberg, 2018"), note="a = 1.5, b = 1, c = 0.5 (example)")
fn("psvelu", "Parametric shifted ELU (vertical)", "adaptive-rectifiers",
   lambda x: 1.5 * x - 0.3 if x >= 0 else 1.5 * math.expm1(x) - 0.3,
   [("(a/b)x + c", "x ≥ 0"), ("a(e<sup>x/b</sup> − 1) + c", "x < 0")],
   "PELU with a trainable vertical shift c.",
   None, True, aka="PSvELU", src=S("Grelsson & Felsberg, 2018"), note="a = 1.5, b = 1, c = −0.3 (example)")
fn("t-swish", "Tunable swish", "adaptive-rectifiers",
   lambda x: x if x >= 0.5 else 1.5 * x * sig(1 * x),
   [("x", "x ≥ c"), ("a x σ(bx)", "x < c")],
   "Identity above a threshold c and a scaled Swish below it, with all three parameters optionally trainable.",
   None, None, aka="T-swish", src=S("Javid et al., 2022"), note="a = 1.5, b = 1, c = 0.5 (example)", mono=False)


def _repsku(x, b=0.0, c=1.0, d=1.0, e=1.0):
    if x < b:
        return 0.0
    t = sgn(x - c) * (abs(x - c) / d) ** e
    return (x - b) * sig(t)


fn("repsu", "Rectified parametric sigmoid unit", "adaptive-rectifiers",
   lambda x: 0.5 * _repsku(x) + 0.5 * (2 * x - _repsku(x) if x >= 0 else 0.0),
   "f(x) = a RePSKU(x) + (1 − a) RePSHU(x)",
   "A trainable mix of two rectified sigmoid-shaped components, generalising a shrinkage function used for image denoising.",
   None, None, aka="RePSU", src=S("Atto et al., 2023"), note="a = 0.5, b = 0, c = 1, d = 1, e = 1 (example)")
fn("pdelu", "Parametric deformable ELU", "adaptive-rectifiers",
   lambda x: x if x >= 0 else 1.0 * ((1 + (1 - 0.9) * x) ** (1 / (1 - 0.9)) - 1),
   [("x", "x ≥ 0"), ("a([1 + (1 − b)x]<sup>1/(1 − b)</sup> − 1)", "x < 0")],
   "An ELU whose exponential is replaced by a deformed (power-law) version, controlled by b, which shifts the mean output toward zero.",
   None, None, aka="PDELU", src=S("Cheng et al., 2020"), note="a = 1, b = 0.9 (recommended)", xr=(-8, 4), mono=False)
fn("pfplus", "Parametric first power linear unit", "adaptive-rectifiers",
   lambda x: 0.2 * x if x >= 0 else 0.2 * x / (1 - 10 * x),
   "f(x) = a x (1 − bx)<sup>H(x) − 1</sup>",
   "Linear for positive inputs and a/(1 − bx)-damped for negative ones, using the Heaviside step H. With a = 0.2, b = 10 it resembles ReLU; with a = 5, b = 0.1 a linear map.",
   None, True, aka="PFPLUS", src=S("Duan et al., 2022"), note="a = 0.2, b = 10", mono=False)
fn("pvlu", "Parametric variational linear unit", "adaptive-rectifiers", lambda x: relu(x) + 0.1 * math.sin(6 * x),
   "f(x) = max(0, x) + a sin(bx)",
   "VLU with a trainable ripple amplitude a and frequency b.",
   None, False, aka="PVLU", src=S("Gupta & Ahuja, 2021"), note="a = 0.1, b = 6 (example)")
fn("autotuning-sigmoid", "Shape-autotuning sigmoid", "adaptive-sigmoidal",
   lambda x: 2 * (1 - math.exp(-2 * x)) / (2 * (1 + math.exp(-2 * x))) if x > -300 else -1.0,
   "f(x) = 2(1 − e<sup>−ax</sup>) / (a(1 + e<sup>−ax</sup>))",
   "One of the earliest adaptive activations: one trainable parameter controls both the slope and the amplitude of a bipolar sigmoid.",
   None, True, src=S("Yamada & Yabuta, 1992"), note="a = 2")
fn("generalized-tanh", "Generalised hyperbolic tangent", "adaptive-sigmoidal",
   lambda x: 1.5 * math.tanh(x / 2),
   "f(x) = a(1 − e<sup>−bx</sup>) / (1 + e<sup>−bx</sup>)",
   "A bipolar sigmoid with trainable amplitude a and slope b.",
   None, True, src=S("Chen & Chang, 1996"), note="a = 1.5, b = 1 (example)", xr=(-8, 8))
fn("trainable-amplitude", "Trainable amplitude", "adaptive-frameworks", lambda x: 2 * math.tanh(x) + 0.3,
   "f(x) = a g(x) + b",
   "Adds a trainable amplitude a and offset b to any activation g, applied per neuron. Plotted for g = tanh.",
   None, True, src=S("Trentin, 2001"), note="g = tanh, a = 2, b = 0.3 (example)")
fn("assf", "Adaptive slope sigmoidal function", "adaptive-sigmoidal", lambda x: sig(2 * x),
   "f(x) = σ(ax)",
   "The logistic sigmoid with one global trainable slope a.",
   None, True, aka="ASSF", src=S("Yamada & Yabuta, 1992; Nawi et al., 2009"), note="a = 2")
fn("svaf", "Slope varying activation function", "adaptive-sigmoidal", lambda x: math.tanh(0.5 * x),
   "f(x) = tanh(ax)",
   "Tanh with one global trainable slope a, trained with a modified backpropagation rule.",
   None, True, aka="SVAF", src=S("Bai, 2009"), note="a = 0.5", xr=(-8, 8))
fn("tanhsoft-1", "TanhSoft-1", "adaptive-self-gated", lambda x: math.tanh(0.5 * x) * softplus(x),
   "f(x) = tanh(ax) · ln(1 + e<sup>x</sup>)",
   "Softplus gated by a trainable-slope tanh, which gives a Swish-like curve with a negative lobe.",
   None, True, src=S("Biswas et al., 2020"), note="a = 0.5 (example)")
fn("tanhsoft-2", "TanhSoft-2", "adaptive-self-gated", lambda x: x * math.tanh(0.5 * math.exp(1 * x)) if x < 700 else x,
   "f(x) = x · tanh(b e<sup>cx</sup>)",
   "TanhExp with trainable scale b and rate c.",
   None, True, src=S("Biswas et al., 2020"), note="b = 0.5, c = 1 (example)")
fn("tanhsoft-3", "TanhSoft-3", "adaptive-self-gated", lambda x: math.log(1 + math.exp(x) * math.tanh(x)) if x < 700 else x,
   "f(x) = ln(1 + e<sup>x</sup> tanh(ax))",
   "A softplus-like curve whose inner term is gated by tanh(ax).",
   None, True, src=S("Biswas et al., 2020"), note="a = 1")
fn("psigmoid", "Parametric sigmoid", "adaptive-sigmoidal", lambda x: 2 * sig(x),
   "f(x) = a σ(bx)",
   "A logistic sigmoid with a trainable amplitude a (per neuron or channel) and a slope b.",
   None, True, aka="psigmoid", src=S("Ying et al., 2021; Özbay & Tezel, 2010"), note="a = 2, b = 1")
fn("psf", "Parametric sigmoid function", "adaptive-sigmoidal", lambda x: sig(x) ** 2,
   "f(x) = 1 / (1 + e<sup>−x</sup>)<sup>m</sup>",
   "A sigmoid raised to a trainable power m, which moves its steepest point and changes its asymmetry.",
   None, True, aka="PSF", src=S("Chandra & Singh, 2004"), note="m = 2", xr=(-6, 8))
fn("stac-tanh", "STAC-tanh", "adaptive-sigmoidal",
   lambda x: math.tanh(-1.5) + 0.1 * (x + 1.5) if x < -1.5 else (math.tanh(x) if x <= 1.5 else math.tanh(1.5) + 0.1 * (x - 1.5)),
   [("tanh(−a) + b(x + a)", "x < −a"), ("tanh(x)", "−a ≤ x ≤ a"), ("tanh(a) + b(x − a)", "x > a")],
   "Slope- and threshold-adaptive tanh: tanh in the middle and straight lines of trainable slope b beyond ±a, so the gradient never vanishes.",
   None, None, src=S("Zhang et al., 2021"), note="a = 1.5, b = 0.1 (example)", xr=(-6, 6))
fn("gra", "Generalised Riccati activation", "adaptive-sigmoidal", lambda x: 1 - 1 / (1 + (1 + x) ** 2),
   "f(x) = 1 − a / (a + (1 + bx)<sup>c</sup>)",
   "A rational sigmoid-like function derived from the Riccati equation, with three trainable shape parameters.",
   None, True, aka="GRA", src=S("Protonotarios et al., 2022"), note="a = 1, b = 1, c = 2", mono=False)
fn("ahaf", "Adaptive hybrid activation function", "adaptive-self-gated", lambda x: 1.5 * x * sig(x),
   "f(x) = a x σ(bx)",
   "Swish with a trainable vertical scale a as well as the slope b.",
   None, True, aka="AHAF", src=S("Bodyanskiy & Kostiuk, 2022"), note="a = 1.5, b = 1")
fn("pssilu", "Parametric shifted SiLU", "adaptive-self-gated", lambda x: x * (sig(x) - 0.25) / (1 - 0.25),
   "f(x) = x (σ(ax) − b) / (1 − b)",
   "SiLU whose gate is shifted down by b and renormalised, which deepens the negative region.",
   None, True, aka="PSSiLU", src=S("Dai et al., 2022"), note="a = 1, b = 0.25 (example)", xr=(-6, 4))
fn("acon-b", "ACON-B", "adaptive-self-gated", lambda x: 0.75 * x * sig(0.75 * x) + 0.25 * x,
   "f(x) = (1 − b) x σ(a(1 − b)x) + bx",
   "A smooth maximum of x and bx (b initialised to 0.25): the ACON family's smooth leaky ReLU.",
   None, True, src=S("Ma et al., 2021"), note="a = 1, b = 0.25 (initial values)")
fn("psgu", "Parameterised self-circulating gating unit", "adaptive-self-gated", lambda x: x * math.tanh(sig(x)),
   "f(x) = x · tanh(a σ(x))",
   "x gated by tanh of a scaled sigmoid; the authors also propose a matching initialisation scheme.",
   None, True, aka="PSGU", src=S("Li et al., 2021"), note="a = 1")
fn("tbsrelu-learnable", "TBSReLU (learnable)", "adaptive-self-gated", lambda x: x * math.tanh(0.5 * math.tanh(x / 2)),
   "f(x) = x · tanh(a (1 − e<sup>−x</sup>) / (1 + e<sup>−x</sup>))",
   "TBSReLU with a trainable scale a inside the tanh (initialised to 0.5).",
   None, True, aka="TBSReLUl", src=S("Mercioni et al., 2020"), note="a = 0.5 (initial value)")
fn("pats", "PATS", "adaptive-self-gated", lambda x: x * math.atan(0.5 * math.pi * sig(x)),
   "f(x) = x · arctan(a π σ(x))",
   "Like PSGU but with arctan and a scale a sampled at random during training.",
   None, True, src=S("Zheng & Wang, 2020"), note="a = 0.5 (example)")
fn("aqulu", "Adaptive quadratic linear unit", "adaptive-self-gated",
   lambda x: x if x >= 1 else (0.5 * x * x + 0.5 * x if x >= -1 else 0.0),
   [("x", "x ≥ (1 − b)/a"), ("a x² + b x", "−b/a ≤ x < (1 − b)/a"), ("0", "x < −b/a")],
   "A quadratic stand-in for SiLU: zero, then a trainable parabola, then the identity.",
   None, True, aka="AQuLU", src=S("Wu et al., 2023"), note="a = 0.5, b = 0.5; middle condition written consistently (the survey's is reversed)", mono=False)
fn("erfact", "ErfAct", "adaptive-self-gated", lambda x: x * math.erf(0.75 * math.exp(0.75 * x)) if x < 700 else x,
   "f(x) = x · erf(a e<sup>bx</sup>)",
   "x gated by erf of a trainable exponential.",
   None, True, src=S("Biswas et al., 2022"), note="a = 0.75, b = 0.75 (example)")
fn("pserf", "Parametric Serf", "adaptive-self-gated", lambda x: x * math.erf(1.5 * softplus(x)),
   "f(x) = x · erf(a ln(1 + e<sup>bx</sup>))",
   "Serf with trainable scales a and b.",
   None, True, aka="pserf", src=S("Biswas et al., 2022"), note="a = 1.5, b = 1 (example)")
fn("swim", "Swim", "adaptive-self-gated", lambda x: x * 0.5 * (1 + 0.5 * x / math.sqrt(1 + x * x)),
   "f(x) = x · ½ (1 + a x / √(1 + x²))",
   "PFLU with a scale a on the algebraic gate (a = 0.5 was used).",
   None, True, src=S("Abdool & Dear, 2023"), note="a = 0.5")
fn("arbf", "Adaptive radial basis function", "adaptive-frameworks", lambda x: math.exp(-(x - 1) ** 2 / (2 * 0.5 ** 2)),
   "f(x) = exp(−(x − a)² / 2b²)",
   "A Gaussian bump with trainable centre a and width b.",
   None, True, aka="ARBF", src=S("Jiang et al., 2021"), note="a = 1, b = 0.5 (example)", xr=(-3, 4), mono=False)
fn("pgelu", "Parametric GELU", "adaptive-self-gated", lambda x: x * Phi(x / 2),
   "f(x) = x · Φ(x / a)",
   "GELU with a global trainable scale a, derived from noise injection (a is the RMS noise).",
   None, True, aka="PGELU", src=S("Duan et al., 2024"), note="a = 2")
fn("pfts", "Parametric flatted-T swish", "adaptive-rectifiers", lambda x: relu(x) * sig(x) - 0.2,
   "f(x) = ReLU(x) · σ(x) + T",
   "Flatted-T swish with a trainable floor T per neuron (initialised to −0.2).",
   None, True, aka="PFTS", src=S("Chieng et al., 2020"), note="T = −0.2 (initial value)")
fn("pfpm", "Parametric flatten-p Mish", "adaptive-rectifiers",
   lambda x: x * math.tanh(softplus(x)) - 0.2 if x >= 0 else -0.2,
   [("x tanh(ln(1 + e<sup>x</sup>)) + p", "x ≥ 0"), ("p", "x < 0")],
   "Mish on the positive side and a trainable flat level p on the negative side.",
   None, True, aka="PFPM", src=S("Mondal & Shrivastava, 2022"), note="p = −0.2 (example)")
fn("geu", "Gaussian error unit", "adaptive-sigmoidal", lambda x: Phi(x / 2),
   "f(x) = Φ(x / a)",
   "The normal CDF with a global trainable width a; multiplying it by x gives PGELU.",
   None, True, aka="GEU", src=S("Duan et al., 2024"), note="a = 2", xr=(-8, 8))
fn("sgt", "Scaled-gamma-tanh", "adaptive-rectifiers", lambda x: x ** 1.5 if x >= 0 else 0.1 * x,
   [("a x<sup>b</sup>", "x ≥ 0"), ("c x<sup>d</sup>", "x < 0")],
   "A piecewise power function with trainable exponents on each side.",
   None, None, aka="SGT", src=S("Khagi & Kwon, 2022"), note="a = 1, b = 1.5, c = 0.1, d = 1 (example)", xr=(-3, 3))
fn("rsign", "RSign", "adaptive-sigmoidal", lambda x: 1.0 if x >= 0.5 else -1.0,
   [("1", "x ≥ a"), ("−1", "x < a")],
   "A sign function with a trainable threshold per channel. Used in ReActNet's binary networks.",
   "{−1, 1}", False, src=S("Liu et al., 2020"), note="a = 0.5 (example)")
fn("p-sig-ramp", "P-SIG-RAMP", "adaptive-sigmoidal",
   lambda x: 0.5 * sig(x) + 0.5 * (1.0 if x >= 1 else (0.5 * x + 0.5 if x > -1 else 0.0)),
   "f(x) = a σ(x) + (1 − a) · ramp<sub>b</sub>(x)",
   "A trainable blend of the logistic sigmoid and a linear ramp of trainable slope b.",
   None, False, src=S("Jie et al., 2021"), note="a = 0.5, b = 0.5", xr=(-6, 6))
fn("laaf", "Locally adaptive activation function", "adaptive-frameworks",
   [("tanh(nax)", lambda x: math.tanh(2 * x)), ("σ(nax)", lambda x: sig(2 * x)), ("ReLU(nax)", lambda x: relu(2 * x))],
   "f(x) = g(n · a x)",
   "Gives any activation g a trainable slope a per neuron, often with a fixed scale factor n to speed up training. Developed for physics-informed networks. Plotted with n·a = 2.",
   "depends on g", None, aka="LAAF", src=S("Jagtap et al., 2020"), mono=True)
fn("pstanh", "Parametric scaled tanh", "adaptive-self-gated", lambda x: x * 0.5 * (1 + math.tanh(x)),
   "f(x) = x · a(1 + tanh(bx))",
   "A Swish-like unit built on tanh; since ½(1 + tanh(x)) = σ(2x), it is Swish with β = 2 when a = ½, b = 1.",
   None, True, aka="PSTanh", src=S("Adu et al., 2021"), note="a = 0.5, b = 1")
fn("ssinh", "Scaled sine-hyperbolic", "adaptive-frameworks", lambda x: math.sinh(0.5 * x),
   "f(x) = a sinh(bx)",
   "A hyperbolic sine with trainable scales: nearly linear near zero and growing exponentially outward.",
   None, True, aka="SSinH", src=S("Husain et al., 2021"), note="a = 1, b = 0.5")
fn("sexp", "Scaled exponential function", "adaptive-frameworks", lambda x: math.expm1(0.5 * x),
   "f(x) = a(e<sup>bx</sup> − 1)",
   "A shifted exponential with trainable scales, zero at the origin.",
   None, True, aka="SExp", src=S("Husain et al., 2021"), note="a = 1, b = 0.5")
fn("lau", "Logmoid activation unit", "adaptive-self-gated", lambda x: x * math.log1p(2 * sig(x)),
   "f(x) = x ln(1 + a σ(bx))",
   "Logish with trainable per-layer parameters a and b (both initialised to 1, which gives Logish).",
   None, True, aka="LAU", src=S("Zhou et al., 2023"), note="a = 2, b = 1 (example)")
fn("coslu", "Cosinu-sigmoidal linear unit", "adaptive-self-gated", lambda x: (x + math.cos(x)) * sig(x),
   "f(x) = (x + a cos(bx)) σ(x)",
   "The cosine counterpart of SinLU: a trainable cosine ripple is added before sigmoid gating.",
   None, True, aka="CosLU", src=S("Pishchik, 2023"), note="a = 1, b = 1")
fn("agumb", "Adaptive Gumbel", "adaptive-sigmoidal", lambda x: 1 - (1 + 0.5 * math.exp(x)) ** (-2) if x < 700 else 1.0,
   "f(x) = 1 − (1 + a e<sup>x</sup>)<sup>−1/a</sup>",
   "A trainable asymmetric sigmoid based on the Gumbel distribution (the same family as Aranda-Ordaz).",
   None, True, aka="AGumb", src=S("Farhadi et al., 2019"), note="a = 0.5")
fn("saaaf", "Shape-autotuning adaptive activation", "adaptive-sigmoidal",
   lambda x: x / (x / 3 + math.exp(-x / 2)) if x > -300 else 0.0,
   "f(x) = x / (x/a + e<sup>−x/b</sup>)",
   "The soft-root-sign shape with trainable a and b (requiring 0 < b/a < e): bounded above by a, with a dip for negative inputs.",
   None, True, aka="SAAAF", src=S("Zhou et al., 2021"), note="a = 3, b = 2", xr=(-6, 12))
fn("fracrelu", "Fractional ReLU", "adaptive-frameworks", lambda x: x ** 0.5 / math.gamma(1.5) if x >= 0 else 0.0,
   "f(x) = x<sup>1 − a</sup> / Γ(2 − a)",
   "The a-th fractional derivative of ReLU: a = 0 gives ReLU, a = 1 the step function, and values in between interpolate. Shown for x ≥ 0.",
   None, None, aka="FracReLU", src=S("Zamora-Esquivel et al., 2021"), note="a = 0.5", xr=(-1, 4))
fn("scaled-softsign", "Scaled softsign", "adaptive-sigmoidal", lambda x: 2 * x / (1 + abs(x)),
   "f(x) = a x / (b + |x|)",
   "Softsign with a trainable amplitude a and transition rate b.",
   None, True, src=S("Pishchik, 2023"), note="a = 2, b = 1", xr=(-8, 8))
fn("s-plus-2l", "Parameterised softplus (s+2L)", "adaptive-rectifiers", lambda x: softplus(x) - 0.5,
   "f(x) = ln(1 + e<sup>x</sup>) − a",
   "Softplus with a trainable downward shift a ∈ [0, 1].",
   None, True, aka="s₊2L", src=S("Vargas et al., 2023"), note="a = 0.5")
fn("uaf", "Universal activation function", "adaptive-frameworks", lambda x: softplus(x + 0.5) - softplus(x - 0.5),
   "f(x) = ln(1 + e<sup>a(x + b) + cx²</sup>) − ln(1 + e<sup>d(x − b)</sup>) + e",
   "A difference of softplus terms with five trainable parameters that can approximate the step, sigmoid, tanh, ReLU, leaky ReLU and Gaussian.",
   None, True, aka="UAF", src=S("Yuen et al., 2021"), note="a = 1, b = 0.5, c = 0, d = 1, e = 0 (a sigmoid-like setting)")
fn("leaf", "Learnable extended activation function", "adaptive-frameworks", lambda x: (x + 0.5) * sig(2 * x) - 0.2,
   "f(x) = (ax + b) σ(cx) + d",
   "Four trainable parameters around a sigmoid gate; particular settings reproduce SiLU, the sigmoid, tanh-like and ReLU-like shapes.",
   None, True, aka="LEAF", src=S("Bodyanskiy & Kostiuk, 2023"), note="a = 1, b = 0.5, c = 2, d = −0.2 (example)", mono=False)
fn("grelu-general", "Generalised ReLU (GReLU)", "adaptive-rectifiers", lambda x: math.log1p(2 ** x) / math.log(2) if x < 1000 else x,
   "f(x) = ln(1 + a<sup>bx</sup>) / (b ln a)",
   "A softplus in base a with a trainable sharpness b; as b grows it approaches ReLU.",
   None, True, aka="GReLU", src=S("Terziyan et al., 2022"), note="a = 2, b = 1")
fn("maf-multiquadratic", "Multiquadratic activation", "adaptive-frameworks", lambda x: math.sqrt((x - 0.5) ** 2 + 1),
   "f(x) = √(‖x − a‖² + b²)",
   "A multiquadric radial basis function with trainable centre a and width b.",
   None, True, aka="MAF", src=S("Huang et al., 2012"), note="a = 0.5, b = 1", mono=False)
fn("eis-1", "EIS-1", "adaptive-frameworks", lambda x: x * softplus(x) / (x + math.exp(-x)) if x > -700 else 0.0,
   "f(x) = x ln(1 + e<sup>x</sup>) / (x + d e<sup>−ex</sup>)",
   "A member of the EIS family (softplus-weighted rational functions) with trainable parameters d and e.",
   None, True, src=S("Biswas et al., 2021"), note="d = 1, e = 1", mono=False)
fn("eis-2", "EIS-2", "adaptive-frameworks", lambda x: x * softplus(x) / math.sqrt(1 + x * x),
   "f(x) = x ln(1 + e<sup>x</sup>) / √(b + cx²)",
   "An EIS family member that normalises x · softplus(x) by an algebraic term, giving roughly linear growth for large inputs.",
   None, True, src=S("Biswas et al., 2021"), note="b = 1, c = 1", mono=False)
fn("eis-3", "EIS-3", "adaptive-frameworks", lambda x: x / (1 + 2 * math.exp(-x)) if x > -700 else 0.0,
   "f(x) = x / (1 + d e<sup>−ex</sup>)",
   "An EIS family member equivalent to a shifted Swish: d sets the offset of the gate and e its slope.",
   None, True, src=S("Biswas et al., 2021"), note="d = 2, e = 1")


# =============================================================================== batch 7
# Adaptive activation functions, part 3 (survey §2.21–2.47): piecewise-linear adaptive units, blending and
# polynomial families, rational units, truncated Gaussian, kernel activations and the MSRF ("Plus") family.

def mollify(v, eps):
    """MSRF smoothing of |v|: √(v² + ε)."""
    return math.sqrt(v * v + eps)


fn("elus-plus-2l", "ELU + parameterised softplus", "adaptive-rectifiers",
   lambda x: 0.5 * elu_(x) + 0.5 * (softplus(x) - 0.5),
   "f(x) = b ELU(x) + (1 − b) s₊2L(x)",
   "A trainable blend of ELU and the shifted softplus s₊2L.",
   None, True, aka="ELUs₊2L", src=S("Vargas et al., 2023"), note="b = 0.5, softplus shift 0.5 (example)")
fn("naf", "Neuron-adaptive activation function", "adaptive-frameworks",
   lambda x: math.exp(-x * x) + 1 / (1 + math.exp(-2 * x)),
   "f(x) = a e<sup>−bx²</sup> + c / (1 + e<sup>−dx</sup>)",
   "A Gaussian bump plus a sigmoid, with four network-wide trainable parameters; it performed well on small regression tasks.",
   None, True, aka="NAF", src=S("Xu & Zhang, 2000"), note="a = 1, b = 1, c = 1, d = 2 (example)", mono=False)
fn("sine-gauss-sigmoid", "Sine–Gaussian–sigmoid combination", "adaptive-frameworks",
   lambda x: 0.5 * math.sin(2 * x) + 0.5 * math.exp(-x * x) + 1 / (1 + math.exp(-2 * x)),
   "f(x) = a sin(bx) + c e<sup>−dx²</sup> + e / (1 + e<sup>−fx</sup>)",
   "A six-parameter trainable mix of a sine, a Gaussian and a scaled sigmoid, an extension of the scaled logistic sigmoid.",
   None, True, src=S("Xu & Zhang, 2002"), note="a = 0.5, b = 2, c = 0.5, d = 1, e = 1, f = 2 (example)", mono=False)
fn("splash", "SPLASH", "adaptive-rectifiers",
   lambda x: 1.0 * relu(x) + 0.5 * relu(x - 1) + 0.2 * relu(-x) + 0.3 * relu(-x - 1),
   "f(x) = Σ<sub>s</sub> a⁺<sub>s</sub> max(0, x − b<sub>s</sub>) + Σ<sub>s</sub> a⁻<sub>s</sub> max(0, −x − b<sub>s</sub>)",
   "Simple piecewise linear and adaptive function with symmetric hinges: hinges at ±b<sub>s</sub> with separate trainable slopes on each side. Seven hinges usually worked well.",
   None, False, src=S("Tavakoli et al., 2021"), note="hinges at 0 and ±1; a⁺ = (1, 0.5), a⁻ = (0.2, 0.3) (example)", mono=False)
fn("melu", "Mexican ReLU", "adaptive-rectifiers",
   lambda x: (x if x >= 0 else 0.25 * x) + 0.5 * max(1 - abs(x - 1), 0.0),
   "f(x) = PReLU(x) + Σ<sub>j</sub> a<sub>j</sub> max(c<sub>j</sub> − |x − b<sub>j</sub>|, 0)",
   "PReLU plus a sum of trainable 'Mexican hat' tent functions at fixed positions, which lets each unit learn bumps in its response.",
   None, False, aka="MeLU", src=S("Maguolo et al., 2021"), note="PReLU slope 0.25; one tent with a = 0.5, b = 1, c = 1 (example)", mono=False)
fn("mmelu", "Modified Mexican ReLU", "adaptive-rectifiers",
   lambda x: 0.5 * max(1 - abs(x - 1), 0.0) + 0.5 * relu(x),
   "f(x) = a max(b − |x − c|, 0) + (1 − a) ReLU(x)",
   "A blend of one tent function and ReLU whose parameters are estimated with a Bayesian procedure.",
   None, False, aka="MMeLU", src=S("Fakhfakh & Chaari, 2023"), note="a = 0.5, b = 1, c = 1 (example)", mono=False)
fn("galu", "Gaussian ReLU", "adaptive-rectifiers",
   lambda x: (x if x >= 0 else 0.25 * x) + 0.5 * (max(1 - abs(x - 1), 0.0) + min(abs(x - 1 - 2) - 1, 0.0)),
   "φ(x) = max(c − |x − b|, 0) + min(|x − b − 2c| − c, 0)",
   "MeLU with components made of a tent followed by an inverted tent, giving alternating bumps and dips.",
   None, False, aka="GaLU", src=S("Nanni et al., 2020"), note="PReLU slope 0.25; a = 0.5, b = 1, c = 1 (example)", xr=(-3, 5), mono=False)
fn("hard-swish-adaptive", "Hard-Swish (adaptive)", "adaptive-rectifiers",
   lambda x: 2 * x * max(0.0, min(0.2 * 1 * x + 0.5, 1.0)),
   "f(x) = 2x · max(0, min(0.2bx + 0.5, 1))",
   "A scaled hard sigmoid gate with a trainable slope b; as b → ∞ it approaches ReLU. Distinct from MobileNetV3's hard swish.",
   None, False, src=S("Avenash & Viswanath, 2019"), note="b = 1", xr=(-4, 4))
fn("n-activation", "N-activation", "adaptive-rectifiers",
   lambda x: x - 2 * (-1) if x < -1 else (-x if x <= 1 else x - 2 * 1),
   [("x − 2t<sub>min</sub>", "x < t<sub>min</sub>"), ("−x", "t<sub>min</sub> ≤ x ≤ t<sub>max</sub>"), ("x − 2t<sub>max</sub>", "x > t<sub>max</sub>")],
   "An N-shaped piecewise-linear function with trainable thresholds, rising, falling and rising again. Proposed for Lipschitz-constrained networks.",
   None, False, src=S("Prach & Lampert, 2023"), note="t_min = −1, t_max = 1", mono=False)
fn("alisa", "ALiSA", "adaptive-rectifiers",
   lambda x: 0.5 * x - 0.5 + 1 if x >= 1 else (x if x > 0 else 0.1 * x),
   [("a<sub>r</sub>x − a<sub>r</sub> + 1", "x ≥ 1"), ("x", "0 < x < 1"), ("a<sub>l</sub>x", "x ≤ 0")],
   "SReLU with its thresholds fixed at 0 and 1 and trainable outer slopes. With fixed slopes it is called LiSA.",
   None, False, src=S("Bawa & Kumar, 2019"), note="a_r = 0.5, a_l = 0.1 (example)")
fn("all-relu", "Alternated left ReLU", "adaptive-rectifiers",
   [("even layers", lambda x: x if x > 0 else -0.1 * x), ("odd layers", lambda x: x if x > 0 else 0.1 * x)],
   [("−ax", "x ≤ 0, even layer"), ("ax", "x ≤ 0, odd layer"), ("x", "x > 0")],
   "A leaky ReLU whose negative slope flips sign from layer to layer. Designed for sparse networks.",
   None, False, aka="All-ReLU", src=S("Curci et al., 2021"), note="a = 0.1", mono=False)
fn("plu", "Piecewise linear unit (PLU)", "adaptive-rectifiers",
   lambda x: max(0.1 * (x + 1) - 1, min(0.1 * (x - 1) + 1, x)),
   "f(x) = max(a(x + b) − b, min(a(x − b) + b, x))",
   "Identity on [−b, b] and slope a outside, like a symmetric SReLU, but always invertible.",
   None, False, aka="PLU", src=S("Nicolae, 2018"), note="a = 0.1, b = 1", xr=(-5, 5))
fn("adalu", "Adaptive linear unit", "adaptive-rectifiers",
   lambda x: x if (x > 0 and x > -0.5) else (0.2 * x if (x <= 0 and 0.2 * x > -0.5) else -0.5),
   [("c(x − a) + b", "x − a > 0 and c(x − a) > e"), ("d(x − a) + b", "x − a ≤ 0 and d(x − a) > e"), ("e + b", "otherwise")],
   "A piecewise-linear unit with trainable offsets, slopes on each side and a saturation level e.",
   None, False, aka="AdaLU", src=S("Mo et al., 2022"), note="a = 0, b = 0, c = 1, d = 0.2, e = −0.5 (example)", xr=(-5, 3))
fn("aria", "ARiA", "adaptive-self-gated", lambda x: x * (1 + math.exp(-2 * x)) ** -1.5 if x > -300 else 0.0,
   "f(x) = x · (1 + e<sup>−bx</sup>)<sup>−a</sup>",
   "Swish with the sigmoid replaced by Richards' curve, a generalised logistic with extra shape parameters (the two-parameter form is plotted).",
   None, True, src=S("Patwardhan et al., 2018"), note="a = 1.5, b = 2 (example)")
fn("modified-weibull", "Modified Weibull function", "adaptive-frameworks",
   lambda x: (x / 1) ** (2 - 1) * math.exp(-(x / 1) ** 2) if x >= 0 else None,
   "f(x) = (x/a)<sup>b − 1</sup> exp(−(x/c)<sup>d</sup>)",
   "A Weibull-shaped bump: polynomial growth for small inputs, exponential decay for large ones, with a trainable peak position. Defined for x ≥ 0.",
   None, True, aka="MWF", src=S("Husain et al., 2021"), note="a = 1, b = 2, c = 1, d = 2 (example)", xr=(0, 3), mono=False)
fn("sincos", "Sincos", "adaptive-frameworks", lambda x: math.sin(x) + 0.5 * math.cos(2 * x),
   "f(x) = a sin(bx) + c cos(dx)",
   "A trainable sum of a sine and a cosine.",
   None, True, src=S("Efe, 2008"), note="a = 1, b = 1, c = 0.5, d = 2 (example)", xr=(-6.3, 6.3))
fn("css", "Sine + sigmoid (CSS)", "adaptive-frameworks", lambda x: 0.5 * math.sin(2 * x) + sig(x),
   "f(x) = a sin(bx) + c σ(dx)",
   "A trainable sum of a sine and a sigmoid.",
   None, True, aka="CSS", src=S("Özbay & Tezel, 2010"), note="a = 0.5, b = 2, c = 1, d = 1 (example)", mono=False)
fn("catalytic-af", "Catalytic activation", "adaptive-frameworks", lambda x: x * math.sin(0.5) + relu(x) * math.cos(0.5),
   "f(x) = x sin(a) + g(x) cos(a)",
   "Mixes the identity and any activation g through a trainable angle a, so a unit can move continuously between linear and nonlinear. Plotted for g = ReLU.",
   None, False, aka="CatAF", src=S("Sarkar et al., 2022"), note="g = ReLU, a = 0.5")
fn("expcos", "Expcos", "adaptive-frameworks", lambda x: math.exp(-0.5 * x * x) * math.cos(3 * x),
   "f(x) = e<sup>−ax²</sup> cos(bx)",
   "A cosine wave under a Gaussian envelope: a localised oscillation with trainable width and frequency.",
   None, True, src=S("Özbay & Tezel, 2010"), note="a = 0.5, b = 3 (example)", xr=(-4, 4), mono=False)
fn("abu", "Adaptive blending unit", "adaptive-frameworks",
   lambda x: 0.5 * relu(x) + 0.3 * math.tanh(x) + 0.2 * x,
   "f(x) = Σ<sub>j</sub> a<sub>j</sub> g<sub>j</sub>(x)",
   "A trainable weighted sum of activations from a fixed pool, learned per layer, so the network can choose its own activation.",
   None, False, aka="ABU", src=S("Sütfeld et al., 2020"), note="0.5 ReLU + 0.3 tanh + 0.2 identity (example)")
fn("tca", "Trainable compound activation", "adaptive-frameworks",
   lambda x: (math.tanh(math.exp(0.3) * x + 0.2) + relu(math.exp(-0.2) * x - 0.3)) / 2,
   "f(x) = (1/k) Σ<sub>j</sub> f<sub>j</sub>(e<sup>a<sub>j</sub></sup> x + b<sub>j</sub>)",
   "Averages several activations, each with its own trainable horizontal scale and shift.",
   None, None, aka="TCA", src=S("Baggenstoss, 2022"), note="tanh and ReLU, with example scales and shifts")
fn("gabu", "Gating adaptive blending unit", "adaptive-frameworks",
   lambda x: sig(1) * relu(x) + sig(-1) * math.tanh(x) + sig(0) * sig(x),
   "f(x) = Σ<sub>j</sub> σ(a<sub>j</sub>) g<sub>j</sub>(x)",
   "Each activation in the pool is switched on or off by a learned sigmoid gate, so every neuron can pick its activation.",
   None, None, aka="GABU", src=S("Dushkoff & Ptucha, 2016"), note="ReLU, tanh and sigmoid with gate logits 1, −1, 0 (example)")
fn("rowdy", "Rowdy activation", "adaptive-frameworks",
   lambda x: relu(x) + 0.1 * 2 * math.sin(1 * 2 * x) + 0.05 * 2 * math.sin(2 * 2 * x),
   "f(x) = a₀ g(x) + Σ<sub>j</sub> a<sub>j</sub> c sin(jcx)",
   "A base activation plus trainable high-frequency sine terms, which helps physics-informed networks escape poor minima. A special case of Deep Kronecker networks.",
   None, False, src=S("Jagtap et al., 2022"), note="g = ReLU, c = 2, a₁ = 0.1, a₂ = 0.05 (example)", mono=False)
fn("slaf", "Self-learnable activation (polynomial)", "adaptive-frameworks",
   lambda x: 0.1 + 0.5 * x + 0.3 * x * x,
   "f(x) = Σ<sub>j</sub> a<sub>j</sub> x<sup>j</sup>",
   "A learnable polynomial per neuron. Inputs are normalised because the gradient grows with powers of x.",
   None, True, aka="SLAF", src=S("Goyal et al., 2019"), note="a = (0.1, 0.5, 0.3) (example)", xr=(-3, 3), mono=False)
fn("chpaf", "Chebyshev polynomial activation", "adaptive-frameworks",
   lambda x: 0.8 * x + 0.2 * (2 * x * x - 1) - 0.1 * (4 * x ** 3 - 3 * x),
   "f(x) = Σ<sub>j</sub> a<sub>j</sub> C<sub>j</sub>(x),  C<sub>j+1</sub> = 2xC<sub>j</sub> − C<sub>j−1</sub>",
   "A learnable combination of Chebyshev polynomials (up to order 3 was used), shared across the network.",
   None, True, aka="ChPAF", src=S("Deepthi et al., 2023"), note="a = (0, 0.8, 0.2, −0.1) (example)", xr=(-1.5, 1.5), mono=False)
fn("lpaf", "Legendre polynomial activation", "adaptive-frameworks",
   lambda x: 0.8 * x + 0.2 * (3 * x * x - 1) / 2 - 0.1 * (5 * x ** 3 - 3 * x) / 2,
   "f(x) = Σ<sub>j</sub> a<sub>j</sub> P<sub>j</sub>(x)",
   "A learnable combination of Legendre polynomials, used to study how well polynomials approximate standard nonlinearities.",
   None, True, aka="LPAF", src=S("Venkatappareddy et al., 2021"), note="a = (0, 0.8, 0.2, −0.1); standard Legendre recursion (the survey's indices are garbled)", xr=(-1.5, 1.5), mono=False)
fn("hpaf", "Hermite polynomial activation", "adaptive-frameworks",
   lambda x: 0.4 * 2 * x + 0.05 * (4 * x * x - 2) - 0.01 * (8 * x ** 3 - 12 * x),
   "f(x) = Σ<sub>j</sub> a<sub>j</sub> H<sub>j</sub>(x)",
   "A learnable combination of (physicists') Hermite polynomials.",
   None, True, aka="HPAF", src=S("Lokhande et al., 2020"), note="a = (0, 0.4, 0.05, −0.01) (example)", xr=(-2.5, 2.5), mono=False)
fn("mogu", "Mixture of Gaussian unit", "adaptive-frameworks",
   lambda x: 1 / math.sqrt(2 * math.pi * 0.49) * math.exp(-x * x / (2 * 0.49)) - 0.5 / math.sqrt(2 * math.pi * 0.25) * math.exp(-(x - 1.5) ** 2 / 0.5),
   "f(x) = Σ<sub>j</sub> a<sub>j</sub> / √(2πσ<sub>j</sub>²) · exp(−(x − μ<sub>j</sub>)² / 2σ<sub>j</sub>²)",
   "A trainable mixture of Gaussian bumps and dips, motivated by the peaks and valleys that look-up-table activations tend to learn.",
   None, True, aka="MoGU", src=S("Wang et al., 2018"), note="two Gaussians: (a, μ, σ) = (1, 0, 0.7) and (−0.5, 1.5, 0.5) (example)", mono=False)
fn("fourier-activation", "Fourier series activation", "adaptive-frameworks",
   lambda x: 0.5 * math.cos(x) + 0.8 * math.sin(x) - 0.2 * math.cos(2 * x) + 0.3 * math.sin(2 * x),
   "f(x) = a + Σ<sub>j=1</sub><sup>r</sup> (b<sub>j</sub> cos(jdx) + c<sub>j</sub> sin(jdx))",
   "A truncated Fourier series with trainable coefficients and base frequency (rank r = 5 was used).",
   None, True, aka="FSA", src=S("Liao, 2020"), note="r = 2, d = 1, example coefficients", xr=(-6.3, 6.3), mono=False)
fn("era", "Enhanced rational activation", "adaptive-frameworks",
   lambda x: (0.02 + 0.5 * x + 0.4 * x * x + 0.1 * x ** 3) / (0.1 + (x * x + 1)),
   "f(x) = Σ<sub>j</sub> a<sub>j</sub>x<sup>j</sup> / (ε + Π<sub>k</sub>((x − c<sub>k</sub>)² + d<sub>k</sub>²))",
   "A rational activation whose denominator is a product of positive quadratics, so it has no poles and needs no absolute value, unlike PAU.",
   None, True, aka="ERA", src=S("Trimmel et al., 2022"), note="cubic numerator, one quadratic factor (c = 0, d = 1, ε = 0.1); example coefficients", mono=False)


def _trug(x, lo, hi, s=1.0):
    a, b = (lo - x) / s, (hi - x) / s
    pdf = lambda v: 0.0 if math.isinf(v) else math.exp(-v * v / 2) / math.sqrt(2 * math.pi)
    upper = lambda v: 0.5 * math.erfc(v / math.sqrt(2))              # 1 − Φ(v)
    denom = (upper(a) - upper(b)) if not math.isinf(b) else upper(a)
    if denom <= 0:
        return None
    return x + s * (pdf(a) - pdf(b)) / denom


fn("trug", "Truncated Gaussian unit", "adaptive-frameworks",
   [("ξ = [0, ∞)", lambda x: _trug(x, 0.0, math.inf)), ("ξ = [−1, 1]", lambda x: _trug(x, -1.0, 1.0))],
   "f(x) = x + σ (φ(α) − φ(β)) / (Φ(β) − Φ(α)),  α = (ξ₁ − x)/σ, β = (ξ₂ − x)/σ",
   "The mean of a Gaussian truncated to [ξ₁, ξ₂]: [0, ∞) gives a smooth ReLU and [−1, 1] a tanh-like curve. The truncation points can be trained.",
   "depends on ξ", True, aka="TruG", src=S("Su et al., 2017"),
   note="σ = 1; standard truncated-normal mean (the survey's denominator has its sign flipped)", mono=True)
fn("kaf", "Kernel activation function", "adaptive-frameworks",
   lambda x: sum(a * math.exp(-(1 / 6) * (x - d) ** 2) for a, d in zip((-0.3, 0.2, 0.8, 1.0, 1.5), (-2, -1, 0, 1, 2))),
   "f(x) = Σ<sub>j</sub> a<sub>j</sub> exp(−γ(x − d<sub>j</sub>)²),  γ = 1/(6Δ²)",
   "A non-parametric activation: a trainable weighted sum of Gaussian kernels centred on a fixed dictionary of points.",
   None, True, aka="KAF", src=S("Scardapane et al., 2019"), note="dictionary −2…2 (Δ = 1), example mixing coefficients", xr=(-5, 5), mono=False)
fn("exp-distance", "Exponential distance activation", "adaptive-frameworks", lambda x: math.exp(-1 * abs(x - 0.5)),
   "f(x) = exp(−a|x − b|)",
   "A Laplacian-kernel bump with trainable sharpness and centre; one of the activations trained in random-weight networks.",
   None, False, src=S("Ertuğrul, 2018"), note="a = 1, b = 0.5 (example)", mono=False)

# ---- the MSRF ('Plus') family: piecewise functions smoothed by |x| → √(x² + ε)
fn("stepplus", "StepPlus", "algebraic", lambda x: 0.5 * (1 + x / mollify(x, 1)),
   "f(x) = ½ (1 + x / √(x² + ε))",
   "A smoothed step function built with the mollified absolute value; its odd counterpart x/√(x² + ε) is called BipolarPlus.",
   None, True, src=S("Pan et al., 2023"), note="ε = 1", xr=(-8, 8))
fn("lreluplus", "LReLUPlus", "algebraic", lambda x: 0.5 * (x + 0.1 * x + mollify(0.9 * x, 0.5)),
   "f(x) = ½ (x + ax + √(((1 − a)x)² + ε))",
   "A smoothed leaky ReLU; also published as SMU-1.",
   None, True, aka="SMU-1", src=S("Pan et al., 2023; Biswas et al., 2022"), note="a = 0.1, ε = 0.5")
fn("softshrinkplus", "SoftshrinkPlus", "algebraic",
   lambda x: x + 0.5 * (math.sqrt((x - 1) ** 2 + 0.1) - math.sqrt((x + 1) ** 2 + 0.1)),
   "f(x) = x + ½ (√((x − a)² + ε) − √((x + a)² + ε))",
   "A smoothed soft-shrink: flat near zero and shifted linear beyond ±a, with rounded corners.",
   None, True, aka="STFPlus", src=S("Pan et al., 2023"), note="a = 1, ε = 0.1")
fn("panplus", "PanPlus", "algebraic", lambda x: -1 + 0.5 * (math.sqrt((x - 1) ** 2 + 0.1) + math.sqrt((x + 1) ** 2 + 0.1)),
   "f(x) = −a + ½ (√((x − a)² + ε) + √((x + a)² + ε))",
   "A smoothed pan function: near zero inside [−a, a], rising linearly outside on both sides.",
   None, True, src=S("Pan et al., 2023"), note="a = 1, ε = 0.1", mono=False)
fn("breluplus", "BReLUPlus", "algebraic", lambda x: 0.5 * (1 + mollify(x, 0.1) - mollify(x - 1, 0.1)),
   "f(x) = ½ (1 + √(x² + ε) − √((x − 1)² + ε))",
   "A smoothed ReLU capped at 1.",
   None, True, src=S("Pan et al., 2023"), note="ε = 0.1", xr=(-3, 4))
fn("sreluplus", "SReLUPlus", "algebraic",
   lambda x: 0.2 * x + 0.5 * (0.2 - 1) * (mollify(x - 1, 0.1) - mollify(x + 1, 0.1)),
   "f(x) = ax + ½(a − 1)(√((x − t)² + ε) − √((x + t)² + ε))",
   "A smoothed symmetric SReLU: slope 1 near zero and slope a beyond ±t.",
   None, True, src=S("Pan et al., 2023"), note="a = 0.2, t = 1, ε = 0.1", xr=(-5, 5))
fn("hardtanhplus", "HardTanhPlus", "algebraic", lambda x: 0.5 * (mollify(x + 1, 0.1) - mollify(x - 1, 0.1)),
   "f(x) = ½ (√((x + 1)² + ε) − √((x − 1)² + ε))",
   "A smoothed HardTanh with rounded corners at ±1.",
   None, True, src=S("Pan et al., 2023"), note="ε = 0.1")
fn("hardshrinkplus", "HardshrinkPlus", "algebraic",
   lambda x: x * (1 + 0.5 * ((x - 1) / math.sqrt((x - 1) ** 2 + 0.1) - (x + 1) / math.sqrt((x + 1) ** 2 + 0.1))),
   "f(x) = x (1 + ½ ((x − a)/√((x − a)² + ε) − (x + a)/√((x + a)² + ε)))",
   "A smoothed hard-shrink: x multiplied by a smooth window that is near zero inside [−a, a].",
   None, True, aka="HTFPlus", src=S("Pan et al., 2023"), note="a = 1, ε = 0.1", xr=(-3, 3), mono=False)
fn("eluplus", "ELUPlus", "algebraic",
   lambda x: 0.5 * (x + mollify(x, 0.1)) + 0.5 * (math.expm1(x) + mollify(math.expm1(x), 0.1)),
   "f(x) = ½(x + |x|<sub>ε</sub>) + ½((e<sup>x</sup> − 1)/a + |(e<sup>x</sup> − 1)/a|<sub>ε</sub>)",
   "The MSRF smoothing applied to ELU's two pieces, as written in the original. Note that, unlike ELU, this form stays positive.",
   None, True, src=S("Pan et al., 2023"), note="a = 1, ε = 0.1; |v|_ε = √(v² + ε)", mono=False)
fn("swishplus", "SwishPlus", "algebraic", lambda x: 0.5 * (x + x * x / mollify(x, 1)),
   "f(x) = x · StepPlus(x) = ½ (x + x² / √(x² + ε))",
   "Swish with the sigmoid replaced by the algebraic StepPlus: no exponentials, same overall shape.",
   None, True, src=S("Pan et al., 2023"), note="ε = 1")
fn("mishplus", "MishPlus", "algebraic", lambda x: x * (lambda u: u / mollify(u, 1))(x / mollify(x, 1)),
   "f(x) = x · BipolarPlus(BipolarPlus(x))",
   "An algebraic Mish-like function built from nested BipolarPlus terms, as written in the original.",
   None, True, src=S("Pan et al., 2023"), note="ε = 1; BipolarPlus(v) = v/√(v² + ε)", mono=False)
fn("logishplus", "LogishPlus", "algebraic", lambda x: x * math.log(1 + 0.5 * (1 + x / mollify(x, 1))),
   "f(x) = x · ln(1 + StepPlus(x))",
   "Logish with the sigmoid replaced by StepPlus.",
   None, True, src=S("Pan et al., 2023"), note="ε = 1")
fn("softsignplus", "SoftsignPlus", "algebraic", lambda x: x / (1 + mollify(x, 0.1)),
   "f(x) = x / (1 + √(x² + ε))",
   "A softsign whose |x| is mollified, making it smooth at zero. With ε = 1 it coincides with rootsig.",
   None, True, src=S("Pan et al., 2023"), note="ε = 0.1", xr=(-8, 8))
fn("signreluplus", "SignReLUPlus", "algebraic",
   lambda x: 0.5 * (x + mollify(x, 0.1)) + (x - mollify(x, 0.1)) / (2 * mollify(1 - x, 0.1)),
   "f(x) = ½(x + |x|<sub>ε</sub>) + (x − |x|<sub>ε</sub>) / (2|1 − x|<sub>ε</sub>)",
   "A smoothed approximation of SignReLU (DLU): ReLU-like for positive inputs and a softsign-like dip for negative ones.",
   None, True, src=S("Pan et al., 2023"), note="ε = 0.1; |v|_ε = √(v² + ε)", xr=(-8, 4))

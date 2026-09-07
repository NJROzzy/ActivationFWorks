# ActivationFWorks

This README catalogs **400 activation functions/variants** used or proposed in neural-network design.
It mixes foundational activations with practical parameterized variants to help with experimentation.

## Why activation functions matter
- They introduce nonlinearity so deep networks can model complex patterns.
- They influence gradient flow, optimization stability, and convergence speed.
- Different tasks (vision, NLP, tabular, implicit fields) can benefit from different shapes.

## 400 activation functions and variants

### Family: Binary Step
- Base idea: Outputs 0/1 based on whether the input crosses a threshold.

1. **Standard Binary Step** — Uses the canonical form with common default parameters.
2. **Scaled Binary Step** — Applies multiplicative gain to control output amplitude.
3. **Shifted Binary Step** — Adds a bias shift to move the operating region.
4. **Temperature Binary Step** — Adjusts sharpness with a temperature parameter.
5. **Clipped Binary Step** — Constrains outputs to a bounded range for stability.
6. **Normalized Binary Step** — Normalizes activation response across channels or batches.
7. **Residual Binary Step** — Combines identity skip with nonlinear activation output.
8. **Adaptive Binary Step** — Learns one or more activation parameters during training.
9. **Sparse Binary Step** — Encourages many near-zero outputs for sparse representations.
10. **Symmetric Binary Step** — Balances positive and negative responses more evenly.
11. **Asymmetric Binary Step** — Uses different behavior on positive versus negative sides.
12. **Noisy Binary Step** — Injects controlled noise to improve exploration and regularization.
13. **Stochastic Binary Step** — Samples activation behavior from a learnable distribution.
14. **Gated Binary Step** — Multiplies activation output by a learned gate.
15. **Damped Binary Step** — Reduces extreme outputs with attenuation at high magnitudes.
16. **Boosted Binary Step** — Amplifies mid-range responses for stronger gradient flow.
17. **Piecewise Binary Step** — Uses segmented linear or nonlinear regions for flexibility.
18. **Quantized Binary Step** — Restricts activation levels for low-precision deployment.
19. **Monotonic Binary Step** — Enforces strictly nondecreasing behavior across the input range.
20. **Periodic Binary Step** — Introduces cyclic response patterns derived from the base form.

### Family: Sigmoid
- Base idea: Smoothly maps inputs to (0,1), often used for probabilities.

21. **Standard Sigmoid** — Uses the canonical form with common default parameters.
22. **Scaled Sigmoid** — Applies multiplicative gain to control output amplitude.
23. **Shifted Sigmoid** — Adds a bias shift to move the operating region.
24. **Temperature Sigmoid** — Adjusts sharpness with a temperature parameter.
25. **Clipped Sigmoid** — Constrains outputs to a bounded range for stability.
26. **Normalized Sigmoid** — Normalizes activation response across channels or batches.
27. **Residual Sigmoid** — Combines identity skip with nonlinear activation output.
28. **Adaptive Sigmoid** — Learns one or more activation parameters during training.
29. **Sparse Sigmoid** — Encourages many near-zero outputs for sparse representations.
30. **Symmetric Sigmoid** — Balances positive and negative responses more evenly.
31. **Asymmetric Sigmoid** — Uses different behavior on positive versus negative sides.
32. **Noisy Sigmoid** — Injects controlled noise to improve exploration and regularization.
33. **Stochastic Sigmoid** — Samples activation behavior from a learnable distribution.
34. **Gated Sigmoid** — Multiplies activation output by a learned gate.
35. **Damped Sigmoid** — Reduces extreme outputs with attenuation at high magnitudes.
36. **Boosted Sigmoid** — Amplifies mid-range responses for stronger gradient flow.
37. **Piecewise Sigmoid** — Uses segmented linear or nonlinear regions for flexibility.
38. **Quantized Sigmoid** — Restricts activation levels for low-precision deployment.
39. **Monotonic Sigmoid** — Enforces strictly nondecreasing behavior across the input range.
40. **Periodic Sigmoid** — Introduces cyclic response patterns derived from the base form.

### Family: Tanh
- Base idea: Zero-centered squashing function that maps inputs to (-1,1).

41. **Standard Tanh** — Uses the canonical form with common default parameters.
42. **Scaled Tanh** — Applies multiplicative gain to control output amplitude.
43. **Shifted Tanh** — Adds a bias shift to move the operating region.
44. **Temperature Tanh** — Adjusts sharpness with a temperature parameter.
45. **Clipped Tanh** — Constrains outputs to a bounded range for stability.
46. **Normalized Tanh** — Normalizes activation response across channels or batches.
47. **Residual Tanh** — Combines identity skip with nonlinear activation output.
48. **Adaptive Tanh** — Learns one or more activation parameters during training.
49. **Sparse Tanh** — Encourages many near-zero outputs for sparse representations.
50. **Symmetric Tanh** — Balances positive and negative responses more evenly.
51. **Asymmetric Tanh** — Uses different behavior on positive versus negative sides.
52. **Noisy Tanh** — Injects controlled noise to improve exploration and regularization.
53. **Stochastic Tanh** — Samples activation behavior from a learnable distribution.
54. **Gated Tanh** — Multiplies activation output by a learned gate.
55. **Damped Tanh** — Reduces extreme outputs with attenuation at high magnitudes.
56. **Boosted Tanh** — Amplifies mid-range responses for stronger gradient flow.
57. **Piecewise Tanh** — Uses segmented linear or nonlinear regions for flexibility.
58. **Quantized Tanh** — Restricts activation levels for low-precision deployment.
59. **Monotonic Tanh** — Enforces strictly nondecreasing behavior across the input range.
60. **Periodic Tanh** — Introduces cyclic response patterns derived from the base form.

### Family: ReLU
- Base idea: Passes positive inputs and zeros out negative inputs.

61. **Standard ReLU** — Uses the canonical form with common default parameters.
62. **Scaled ReLU** — Applies multiplicative gain to control output amplitude.
63. **Shifted ReLU** — Adds a bias shift to move the operating region.
64. **Temperature ReLU** — Adjusts sharpness with a temperature parameter.
65. **Clipped ReLU** — Constrains outputs to a bounded range for stability.
66. **Normalized ReLU** — Normalizes activation response across channels or batches.
67. **Residual ReLU** — Combines identity skip with nonlinear activation output.
68. **Adaptive ReLU** — Learns one or more activation parameters during training.
69. **Sparse ReLU** — Encourages many near-zero outputs for sparse representations.
70. **Symmetric ReLU** — Balances positive and negative responses more evenly.
71. **Asymmetric ReLU** — Uses different behavior on positive versus negative sides.
72. **Noisy ReLU** — Injects controlled noise to improve exploration and regularization.
73. **Stochastic ReLU** — Samples activation behavior from a learnable distribution.
74. **Gated ReLU** — Multiplies activation output by a learned gate.
75. **Damped ReLU** — Reduces extreme outputs with attenuation at high magnitudes.
76. **Boosted ReLU** — Amplifies mid-range responses for stronger gradient flow.
77. **Piecewise ReLU** — Uses segmented linear or nonlinear regions for flexibility.
78. **Quantized ReLU** — Restricts activation levels for low-precision deployment.
79. **Monotonic ReLU** — Enforces strictly nondecreasing behavior across the input range.
80. **Periodic ReLU** — Introduces cyclic response patterns derived from the base form.

### Family: Leaky ReLU
- Base idea: Keeps a small negative slope to reduce dead neurons.

81. **Standard Leaky ReLU** — Uses the canonical form with common default parameters.
82. **Scaled Leaky ReLU** — Applies multiplicative gain to control output amplitude.
83. **Shifted Leaky ReLU** — Adds a bias shift to move the operating region.
84. **Temperature Leaky ReLU** — Adjusts sharpness with a temperature parameter.
85. **Clipped Leaky ReLU** — Constrains outputs to a bounded range for stability.
86. **Normalized Leaky ReLU** — Normalizes activation response across channels or batches.
87. **Residual Leaky ReLU** — Combines identity skip with nonlinear activation output.
88. **Adaptive Leaky ReLU** — Learns one or more activation parameters during training.
89. **Sparse Leaky ReLU** — Encourages many near-zero outputs for sparse representations.
90. **Symmetric Leaky ReLU** — Balances positive and negative responses more evenly.
91. **Asymmetric Leaky ReLU** — Uses different behavior on positive versus negative sides.
92. **Noisy Leaky ReLU** — Injects controlled noise to improve exploration and regularization.
93. **Stochastic Leaky ReLU** — Samples activation behavior from a learnable distribution.
94. **Gated Leaky ReLU** — Multiplies activation output by a learned gate.
95. **Damped Leaky ReLU** — Reduces extreme outputs with attenuation at high magnitudes.
96. **Boosted Leaky ReLU** — Amplifies mid-range responses for stronger gradient flow.
97. **Piecewise Leaky ReLU** — Uses segmented linear or nonlinear regions for flexibility.
98. **Quantized Leaky ReLU** — Restricts activation levels for low-precision deployment.
99. **Monotonic Leaky ReLU** — Enforces strictly nondecreasing behavior across the input range.
100. **Periodic Leaky ReLU** — Introduces cyclic response patterns derived from the base form.

### Family: PReLU
- Base idea: Learns the negative slope during training.

101. **Standard PReLU** — Uses the canonical form with common default parameters.
102. **Scaled PReLU** — Applies multiplicative gain to control output amplitude.
103. **Shifted PReLU** — Adds a bias shift to move the operating region.
104. **Temperature PReLU** — Adjusts sharpness with a temperature parameter.
105. **Clipped PReLU** — Constrains outputs to a bounded range for stability.
106. **Normalized PReLU** — Normalizes activation response across channels or batches.
107. **Residual PReLU** — Combines identity skip with nonlinear activation output.
108. **Adaptive PReLU** — Learns one or more activation parameters during training.
109. **Sparse PReLU** — Encourages many near-zero outputs for sparse representations.
110. **Symmetric PReLU** — Balances positive and negative responses more evenly.
111. **Asymmetric PReLU** — Uses different behavior on positive versus negative sides.
112. **Noisy PReLU** — Injects controlled noise to improve exploration and regularization.
113. **Stochastic PReLU** — Samples activation behavior from a learnable distribution.
114. **Gated PReLU** — Multiplies activation output by a learned gate.
115. **Damped PReLU** — Reduces extreme outputs with attenuation at high magnitudes.
116. **Boosted PReLU** — Amplifies mid-range responses for stronger gradient flow.
117. **Piecewise PReLU** — Uses segmented linear or nonlinear regions for flexibility.
118. **Quantized PReLU** — Restricts activation levels for low-precision deployment.
119. **Monotonic PReLU** — Enforces strictly nondecreasing behavior across the input range.
120. **Periodic PReLU** — Introduces cyclic response patterns derived from the base form.

### Family: ELU
- Base idea: Uses an exponential curve for negative inputs to improve mean activations.

121. **Standard ELU** — Uses the canonical form with common default parameters.
122. **Scaled ELU** — Applies multiplicative gain to control output amplitude.
123. **Shifted ELU** — Adds a bias shift to move the operating region.
124. **Temperature ELU** — Adjusts sharpness with a temperature parameter.
125. **Clipped ELU** — Constrains outputs to a bounded range for stability.
126. **Normalized ELU** — Normalizes activation response across channels or batches.
127. **Residual ELU** — Combines identity skip with nonlinear activation output.
128. **Adaptive ELU** — Learns one or more activation parameters during training.
129. **Sparse ELU** — Encourages many near-zero outputs for sparse representations.
130. **Symmetric ELU** — Balances positive and negative responses more evenly.
131. **Asymmetric ELU** — Uses different behavior on positive versus negative sides.
132. **Noisy ELU** — Injects controlled noise to improve exploration and regularization.
133. **Stochastic ELU** — Samples activation behavior from a learnable distribution.
134. **Gated ELU** — Multiplies activation output by a learned gate.
135. **Damped ELU** — Reduces extreme outputs with attenuation at high magnitudes.
136. **Boosted ELU** — Amplifies mid-range responses for stronger gradient flow.
137. **Piecewise ELU** — Uses segmented linear or nonlinear regions for flexibility.
138. **Quantized ELU** — Restricts activation levels for low-precision deployment.
139. **Monotonic ELU** — Enforces strictly nondecreasing behavior across the input range.
140. **Periodic ELU** — Introduces cyclic response patterns derived from the base form.

### Family: SELU
- Base idea: Self-normalizing ELU variant with fixed scaling constants.

141. **Standard SELU** — Uses the canonical form with common default parameters.
142. **Scaled SELU** — Applies multiplicative gain to control output amplitude.
143. **Shifted SELU** — Adds a bias shift to move the operating region.
144. **Temperature SELU** — Adjusts sharpness with a temperature parameter.
145. **Clipped SELU** — Constrains outputs to a bounded range for stability.
146. **Normalized SELU** — Normalizes activation response across channels or batches.
147. **Residual SELU** — Combines identity skip with nonlinear activation output.
148. **Adaptive SELU** — Learns one or more activation parameters during training.
149. **Sparse SELU** — Encourages many near-zero outputs for sparse representations.
150. **Symmetric SELU** — Balances positive and negative responses more evenly.
151. **Asymmetric SELU** — Uses different behavior on positive versus negative sides.
152. **Noisy SELU** — Injects controlled noise to improve exploration and regularization.
153. **Stochastic SELU** — Samples activation behavior from a learnable distribution.
154. **Gated SELU** — Multiplies activation output by a learned gate.
155. **Damped SELU** — Reduces extreme outputs with attenuation at high magnitudes.
156. **Boosted SELU** — Amplifies mid-range responses for stronger gradient flow.
157. **Piecewise SELU** — Uses segmented linear or nonlinear regions for flexibility.
158. **Quantized SELU** — Restricts activation levels for low-precision deployment.
159. **Monotonic SELU** — Enforces strictly nondecreasing behavior across the input range.
160. **Periodic SELU** — Introduces cyclic response patterns derived from the base form.

### Family: Softplus
- Base idea: Smooth approximation of ReLU using log(1+exp(x)).

161. **Standard Softplus** — Uses the canonical form with common default parameters.
162. **Scaled Softplus** — Applies multiplicative gain to control output amplitude.
163. **Shifted Softplus** — Adds a bias shift to move the operating region.
164. **Temperature Softplus** — Adjusts sharpness with a temperature parameter.
165. **Clipped Softplus** — Constrains outputs to a bounded range for stability.
166. **Normalized Softplus** — Normalizes activation response across channels or batches.
167. **Residual Softplus** — Combines identity skip with nonlinear activation output.
168. **Adaptive Softplus** — Learns one or more activation parameters during training.
169. **Sparse Softplus** — Encourages many near-zero outputs for sparse representations.
170. **Symmetric Softplus** — Balances positive and negative responses more evenly.
171. **Asymmetric Softplus** — Uses different behavior on positive versus negative sides.
172. **Noisy Softplus** — Injects controlled noise to improve exploration and regularization.
173. **Stochastic Softplus** — Samples activation behavior from a learnable distribution.
174. **Gated Softplus** — Multiplies activation output by a learned gate.
175. **Damped Softplus** — Reduces extreme outputs with attenuation at high magnitudes.
176. **Boosted Softplus** — Amplifies mid-range responses for stronger gradient flow.
177. **Piecewise Softplus** — Uses segmented linear or nonlinear regions for flexibility.
178. **Quantized Softplus** — Restricts activation levels for low-precision deployment.
179. **Monotonic Softplus** — Enforces strictly nondecreasing behavior across the input range.
180. **Periodic Softplus** — Introduces cyclic response patterns derived from the base form.

### Family: Softsign
- Base idea: Squashes using x/(1+|x|), with gentler tails than tanh.

181. **Standard Softsign** — Uses the canonical form with common default parameters.
182. **Scaled Softsign** — Applies multiplicative gain to control output amplitude.
183. **Shifted Softsign** — Adds a bias shift to move the operating region.
184. **Temperature Softsign** — Adjusts sharpness with a temperature parameter.
185. **Clipped Softsign** — Constrains outputs to a bounded range for stability.
186. **Normalized Softsign** — Normalizes activation response across channels or batches.
187. **Residual Softsign** — Combines identity skip with nonlinear activation output.
188. **Adaptive Softsign** — Learns one or more activation parameters during training.
189. **Sparse Softsign** — Encourages many near-zero outputs for sparse representations.
190. **Symmetric Softsign** — Balances positive and negative responses more evenly.
191. **Asymmetric Softsign** — Uses different behavior on positive versus negative sides.
192. **Noisy Softsign** — Injects controlled noise to improve exploration and regularization.
193. **Stochastic Softsign** — Samples activation behavior from a learnable distribution.
194. **Gated Softsign** — Multiplies activation output by a learned gate.
195. **Damped Softsign** — Reduces extreme outputs with attenuation at high magnitudes.
196. **Boosted Softsign** — Amplifies mid-range responses for stronger gradient flow.
197. **Piecewise Softsign** — Uses segmented linear or nonlinear regions for flexibility.
198. **Quantized Softsign** — Restricts activation levels for low-precision deployment.
199. **Monotonic Softsign** — Enforces strictly nondecreasing behavior across the input range.
200. **Periodic Softsign** — Introduces cyclic response patterns derived from the base form.

### Family: Swish
- Base idea: Uses x*sigmoid(x) for smooth non-monotonic behavior.

201. **Standard Swish** — Uses the canonical form with common default parameters.
202. **Scaled Swish** — Applies multiplicative gain to control output amplitude.
203. **Shifted Swish** — Adds a bias shift to move the operating region.
204. **Temperature Swish** — Adjusts sharpness with a temperature parameter.
205. **Clipped Swish** — Constrains outputs to a bounded range for stability.
206. **Normalized Swish** — Normalizes activation response across channels or batches.
207. **Residual Swish** — Combines identity skip with nonlinear activation output.
208. **Adaptive Swish** — Learns one or more activation parameters during training.
209. **Sparse Swish** — Encourages many near-zero outputs for sparse representations.
210. **Symmetric Swish** — Balances positive and negative responses more evenly.
211. **Asymmetric Swish** — Uses different behavior on positive versus negative sides.
212. **Noisy Swish** — Injects controlled noise to improve exploration and regularization.
213. **Stochastic Swish** — Samples activation behavior from a learnable distribution.
214. **Gated Swish** — Multiplies activation output by a learned gate.
215. **Damped Swish** — Reduces extreme outputs with attenuation at high magnitudes.
216. **Boosted Swish** — Amplifies mid-range responses for stronger gradient flow.
217. **Piecewise Swish** — Uses segmented linear or nonlinear regions for flexibility.
218. **Quantized Swish** — Restricts activation levels for low-precision deployment.
219. **Monotonic Swish** — Enforces strictly nondecreasing behavior across the input range.
220. **Periodic Swish** — Introduces cyclic response patterns derived from the base form.

### Family: Mish
- Base idea: Uses x*tanh(softplus(x)) for smooth self-regularization.

221. **Standard Mish** — Uses the canonical form with common default parameters.
222. **Scaled Mish** — Applies multiplicative gain to control output amplitude.
223. **Shifted Mish** — Adds a bias shift to move the operating region.
224. **Temperature Mish** — Adjusts sharpness with a temperature parameter.
225. **Clipped Mish** — Constrains outputs to a bounded range for stability.
226. **Normalized Mish** — Normalizes activation response across channels or batches.
227. **Residual Mish** — Combines identity skip with nonlinear activation output.
228. **Adaptive Mish** — Learns one or more activation parameters during training.
229. **Sparse Mish** — Encourages many near-zero outputs for sparse representations.
230. **Symmetric Mish** — Balances positive and negative responses more evenly.
231. **Asymmetric Mish** — Uses different behavior on positive versus negative sides.
232. **Noisy Mish** — Injects controlled noise to improve exploration and regularization.
233. **Stochastic Mish** — Samples activation behavior from a learnable distribution.
234. **Gated Mish** — Multiplies activation output by a learned gate.
235. **Damped Mish** — Reduces extreme outputs with attenuation at high magnitudes.
236. **Boosted Mish** — Amplifies mid-range responses for stronger gradient flow.
237. **Piecewise Mish** — Uses segmented linear or nonlinear regions for flexibility.
238. **Quantized Mish** — Restricts activation levels for low-precision deployment.
239. **Monotonic Mish** — Enforces strictly nondecreasing behavior across the input range.
240. **Periodic Mish** — Introduces cyclic response patterns derived from the base form.

### Family: GELU
- Base idea: Weights inputs by Gaussian probability mass for transformer models.

241. **Standard GELU** — Uses the canonical form with common default parameters.
242. **Scaled GELU** — Applies multiplicative gain to control output amplitude.
243. **Shifted GELU** — Adds a bias shift to move the operating region.
244. **Temperature GELU** — Adjusts sharpness with a temperature parameter.
245. **Clipped GELU** — Constrains outputs to a bounded range for stability.
246. **Normalized GELU** — Normalizes activation response across channels or batches.
247. **Residual GELU** — Combines identity skip with nonlinear activation output.
248. **Adaptive GELU** — Learns one or more activation parameters during training.
249. **Sparse GELU** — Encourages many near-zero outputs for sparse representations.
250. **Symmetric GELU** — Balances positive and negative responses more evenly.
251. **Asymmetric GELU** — Uses different behavior on positive versus negative sides.
252. **Noisy GELU** — Injects controlled noise to improve exploration and regularization.
253. **Stochastic GELU** — Samples activation behavior from a learnable distribution.
254. **Gated GELU** — Multiplies activation output by a learned gate.
255. **Damped GELU** — Reduces extreme outputs with attenuation at high magnitudes.
256. **Boosted GELU** — Amplifies mid-range responses for stronger gradient flow.
257. **Piecewise GELU** — Uses segmented linear or nonlinear regions for flexibility.
258. **Quantized GELU** — Restricts activation levels for low-precision deployment.
259. **Monotonic GELU** — Enforces strictly nondecreasing behavior across the input range.
260. **Periodic GELU** — Introduces cyclic response patterns derived from the base form.

### Family: Hard Sigmoid
- Base idea: Piecewise-linear approximation to sigmoid for efficiency.

261. **Standard Hard Sigmoid** — Uses the canonical form with common default parameters.
262. **Scaled Hard Sigmoid** — Applies multiplicative gain to control output amplitude.
263. **Shifted Hard Sigmoid** — Adds a bias shift to move the operating region.
264. **Temperature Hard Sigmoid** — Adjusts sharpness with a temperature parameter.
265. **Clipped Hard Sigmoid** — Constrains outputs to a bounded range for stability.
266. **Normalized Hard Sigmoid** — Normalizes activation response across channels or batches.
267. **Residual Hard Sigmoid** — Combines identity skip with nonlinear activation output.
268. **Adaptive Hard Sigmoid** — Learns one or more activation parameters during training.
269. **Sparse Hard Sigmoid** — Encourages many near-zero outputs for sparse representations.
270. **Symmetric Hard Sigmoid** — Balances positive and negative responses more evenly.
271. **Asymmetric Hard Sigmoid** — Uses different behavior on positive versus negative sides.
272. **Noisy Hard Sigmoid** — Injects controlled noise to improve exploration and regularization.
273. **Stochastic Hard Sigmoid** — Samples activation behavior from a learnable distribution.
274. **Gated Hard Sigmoid** — Multiplies activation output by a learned gate.
275. **Damped Hard Sigmoid** — Reduces extreme outputs with attenuation at high magnitudes.
276. **Boosted Hard Sigmoid** — Amplifies mid-range responses for stronger gradient flow.
277. **Piecewise Hard Sigmoid** — Uses segmented linear or nonlinear regions for flexibility.
278. **Quantized Hard Sigmoid** — Restricts activation levels for low-precision deployment.
279. **Monotonic Hard Sigmoid** — Enforces strictly nondecreasing behavior across the input range.
280. **Periodic Hard Sigmoid** — Introduces cyclic response patterns derived from the base form.

### Family: Hard Tanh
- Base idea: Piecewise-linear approximation to tanh with clipping.

281. **Standard Hard Tanh** — Uses the canonical form with common default parameters.
282. **Scaled Hard Tanh** — Applies multiplicative gain to control output amplitude.
283. **Shifted Hard Tanh** — Adds a bias shift to move the operating region.
284. **Temperature Hard Tanh** — Adjusts sharpness with a temperature parameter.
285. **Clipped Hard Tanh** — Constrains outputs to a bounded range for stability.
286. **Normalized Hard Tanh** — Normalizes activation response across channels or batches.
287. **Residual Hard Tanh** — Combines identity skip with nonlinear activation output.
288. **Adaptive Hard Tanh** — Learns one or more activation parameters during training.
289. **Sparse Hard Tanh** — Encourages many near-zero outputs for sparse representations.
290. **Symmetric Hard Tanh** — Balances positive and negative responses more evenly.
291. **Asymmetric Hard Tanh** — Uses different behavior on positive versus negative sides.
292. **Noisy Hard Tanh** — Injects controlled noise to improve exploration and regularization.
293. **Stochastic Hard Tanh** — Samples activation behavior from a learnable distribution.
294. **Gated Hard Tanh** — Multiplies activation output by a learned gate.
295. **Damped Hard Tanh** — Reduces extreme outputs with attenuation at high magnitudes.
296. **Boosted Hard Tanh** — Amplifies mid-range responses for stronger gradient flow.
297. **Piecewise Hard Tanh** — Uses segmented linear or nonlinear regions for flexibility.
298. **Quantized Hard Tanh** — Restricts activation levels for low-precision deployment.
299. **Monotonic Hard Tanh** — Enforces strictly nondecreasing behavior across the input range.
300. **Periodic Hard Tanh** — Introduces cyclic response patterns derived from the base form.

### Family: Hard Swish
- Base idea: Efficient approximation to swish used in mobile architectures.

301. **Standard Hard Swish** — Uses the canonical form with common default parameters.
302. **Scaled Hard Swish** — Applies multiplicative gain to control output amplitude.
303. **Shifted Hard Swish** — Adds a bias shift to move the operating region.
304. **Temperature Hard Swish** — Adjusts sharpness with a temperature parameter.
305. **Clipped Hard Swish** — Constrains outputs to a bounded range for stability.
306. **Normalized Hard Swish** — Normalizes activation response across channels or batches.
307. **Residual Hard Swish** — Combines identity skip with nonlinear activation output.
308. **Adaptive Hard Swish** — Learns one or more activation parameters during training.
309. **Sparse Hard Swish** — Encourages many near-zero outputs for sparse representations.
310. **Symmetric Hard Swish** — Balances positive and negative responses more evenly.
311. **Asymmetric Hard Swish** — Uses different behavior on positive versus negative sides.
312. **Noisy Hard Swish** — Injects controlled noise to improve exploration and regularization.
313. **Stochastic Hard Swish** — Samples activation behavior from a learnable distribution.
314. **Gated Hard Swish** — Multiplies activation output by a learned gate.
315. **Damped Hard Swish** — Reduces extreme outputs with attenuation at high magnitudes.
316. **Boosted Hard Swish** — Amplifies mid-range responses for stronger gradient flow.
317. **Piecewise Hard Swish** — Uses segmented linear or nonlinear regions for flexibility.
318. **Quantized Hard Swish** — Restricts activation levels for low-precision deployment.
319. **Monotonic Hard Swish** — Enforces strictly nondecreasing behavior across the input range.
320. **Periodic Hard Swish** — Introduces cyclic response patterns derived from the base form.

### Family: Sine
- Base idea: Periodic activation used in implicit neural representations.

321. **Standard Sine** — Uses the canonical form with common default parameters.
322. **Scaled Sine** — Applies multiplicative gain to control output amplitude.
323. **Shifted Sine** — Adds a bias shift to move the operating region.
324. **Temperature Sine** — Adjusts sharpness with a temperature parameter.
325. **Clipped Sine** — Constrains outputs to a bounded range for stability.
326. **Normalized Sine** — Normalizes activation response across channels or batches.
327. **Residual Sine** — Combines identity skip with nonlinear activation output.
328. **Adaptive Sine** — Learns one or more activation parameters during training.
329. **Sparse Sine** — Encourages many near-zero outputs for sparse representations.
330. **Symmetric Sine** — Balances positive and negative responses more evenly.
331. **Asymmetric Sine** — Uses different behavior on positive versus negative sides.
332. **Noisy Sine** — Injects controlled noise to improve exploration and regularization.
333. **Stochastic Sine** — Samples activation behavior from a learnable distribution.
334. **Gated Sine** — Multiplies activation output by a learned gate.
335. **Damped Sine** — Reduces extreme outputs with attenuation at high magnitudes.
336. **Boosted Sine** — Amplifies mid-range responses for stronger gradient flow.
337. **Piecewise Sine** — Uses segmented linear or nonlinear regions for flexibility.
338. **Quantized Sine** — Restricts activation levels for low-precision deployment.
339. **Monotonic Sine** — Enforces strictly nondecreasing behavior across the input range.
340. **Periodic Sine** — Introduces cyclic response patterns derived from the base form.

### Family: Gaussian
- Base idea: Bell-shaped radial activation centered at zero.

341. **Standard Gaussian** — Uses the canonical form with common default parameters.
342. **Scaled Gaussian** — Applies multiplicative gain to control output amplitude.
343. **Shifted Gaussian** — Adds a bias shift to move the operating region.
344. **Temperature Gaussian** — Adjusts sharpness with a temperature parameter.
345. **Clipped Gaussian** — Constrains outputs to a bounded range for stability.
346. **Normalized Gaussian** — Normalizes activation response across channels or batches.
347. **Residual Gaussian** — Combines identity skip with nonlinear activation output.
348. **Adaptive Gaussian** — Learns one or more activation parameters during training.
349. **Sparse Gaussian** — Encourages many near-zero outputs for sparse representations.
350. **Symmetric Gaussian** — Balances positive and negative responses more evenly.
351. **Asymmetric Gaussian** — Uses different behavior on positive versus negative sides.
352. **Noisy Gaussian** — Injects controlled noise to improve exploration and regularization.
353. **Stochastic Gaussian** — Samples activation behavior from a learnable distribution.
354. **Gated Gaussian** — Multiplies activation output by a learned gate.
355. **Damped Gaussian** — Reduces extreme outputs with attenuation at high magnitudes.
356. **Boosted Gaussian** — Amplifies mid-range responses for stronger gradient flow.
357. **Piecewise Gaussian** — Uses segmented linear or nonlinear regions for flexibility.
358. **Quantized Gaussian** — Restricts activation levels for low-precision deployment.
359. **Monotonic Gaussian** — Enforces strictly nondecreasing behavior across the input range.
360. **Periodic Gaussian** — Introduces cyclic response patterns derived from the base form.

### Family: ISRU
- Base idea: Inverse square root unit that smoothly compresses large magnitudes.

361. **Standard ISRU** — Uses the canonical form with common default parameters.
362. **Scaled ISRU** — Applies multiplicative gain to control output amplitude.
363. **Shifted ISRU** — Adds a bias shift to move the operating region.
364. **Temperature ISRU** — Adjusts sharpness with a temperature parameter.
365. **Clipped ISRU** — Constrains outputs to a bounded range for stability.
366. **Normalized ISRU** — Normalizes activation response across channels or batches.
367. **Residual ISRU** — Combines identity skip with nonlinear activation output.
368. **Adaptive ISRU** — Learns one or more activation parameters during training.
369. **Sparse ISRU** — Encourages many near-zero outputs for sparse representations.
370. **Symmetric ISRU** — Balances positive and negative responses more evenly.
371. **Asymmetric ISRU** — Uses different behavior on positive versus negative sides.
372. **Noisy ISRU** — Injects controlled noise to improve exploration and regularization.
373. **Stochastic ISRU** — Samples activation behavior from a learnable distribution.
374. **Gated ISRU** — Multiplies activation output by a learned gate.
375. **Damped ISRU** — Reduces extreme outputs with attenuation at high magnitudes.
376. **Boosted ISRU** — Amplifies mid-range responses for stronger gradient flow.
377. **Piecewise ISRU** — Uses segmented linear or nonlinear regions for flexibility.
378. **Quantized ISRU** — Restricts activation levels for low-precision deployment.
379. **Monotonic ISRU** — Enforces strictly nondecreasing behavior across the input range.
380. **Periodic ISRU** — Introduces cyclic response patterns derived from the base form.

### Family: Bent Identity
- Base idea: Near-linear activation with mild nonlinearity around zero.

381. **Standard Bent Identity** — Uses the canonical form with common default parameters.
382. **Scaled Bent Identity** — Applies multiplicative gain to control output amplitude.
383. **Shifted Bent Identity** — Adds a bias shift to move the operating region.
384. **Temperature Bent Identity** — Adjusts sharpness with a temperature parameter.
385. **Clipped Bent Identity** — Constrains outputs to a bounded range for stability.
386. **Normalized Bent Identity** — Normalizes activation response across channels or batches.
387. **Residual Bent Identity** — Combines identity skip with nonlinear activation output.
388. **Adaptive Bent Identity** — Learns one or more activation parameters during training.
389. **Sparse Bent Identity** — Encourages many near-zero outputs for sparse representations.
390. **Symmetric Bent Identity** — Balances positive and negative responses more evenly.
391. **Asymmetric Bent Identity** — Uses different behavior on positive versus negative sides.
392. **Noisy Bent Identity** — Injects controlled noise to improve exploration and regularization.
393. **Stochastic Bent Identity** — Samples activation behavior from a learnable distribution.
394. **Gated Bent Identity** — Multiplies activation output by a learned gate.
395. **Damped Bent Identity** — Reduces extreme outputs with attenuation at high magnitudes.
396. **Boosted Bent Identity** — Amplifies mid-range responses for stronger gradient flow.
397. **Piecewise Bent Identity** — Uses segmented linear or nonlinear regions for flexibility.
398. **Quantized Bent Identity** — Restricts activation levels for low-precision deployment.
399. **Monotonic Bent Identity** — Enforces strictly nondecreasing behavior across the input range.
400. **Periodic Bent Identity** — Introduces cyclic response patterns derived from the base form.

## Notes
- Some entries are canonical named activations, while others are structured variants used for tuning behavior.
- In practice, start with ReLU/GELU/SiLU (Swish) baselines, then test variants if optimization stalls.
- For edge devices, hard/quantized versions are often preferred for efficiency.

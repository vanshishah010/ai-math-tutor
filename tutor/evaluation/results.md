# AI Math Tutor Evaluation Report

**Total Test Problems Evaluated:** 10

## Overall Benchmark Comparison

| Mode | Correct / Total | Overall Accuracy (95% CI) | Format Compliance (%) | huggingface_gsm8k Acc (95% CI) | huggingface_metamath Acc (95% CI) | sympy Acc (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SYMPY** | 3 / 10 | **30.00%** [10.8%, 60.3%] | 30.00% (3/10) | 0.0% [0.0%, 39.0%] | 0.0% [0.0%, 79.3%] | 100.0% [43.9%, 100.0%] |

*Note: All confidence intervals are 95% Wilson score intervals.*

## Side-by-Side Qualitative Examples (5 Selected)

### Example 1 — [huggingface_gsm8k]

**Problem:**
> Alex is saving up for a new car. He already has $14,500 saved up and the car costs $14,600. He decides that to make the last of the money he will deliver groceries for people in the neighborhood.  He charges $1.5 per trip and then 5% of the price of the groceries. If he makes 40 trips and has saved up enough, how many dollars' worth of groceries did he deliver?

**Expected Ground Truth Answer:** `800`

| Evaluation Mode | Model Predicted Answer | Format Compliant? | SymPy Verified? |
| :--- | :--- | :---: | :---: |
| SYMPY | `(no answer extracted)` | No | FAILED (Incorrect) |

<details>
<summary>View SYMPY Raw Output</summary>

```
Could not parse or solve problem.
```
</details>

---

### Example 2 — [huggingface_gsm8k]

**Problem:**
> Elvis is releasing a new album with 10 songs, but he doesn't want to waste too much time on writing. He spends 5 hours in the studio, where he writes, records, and edits his songs. Each song takes 12 minutes to record, then it takes 30 minutes to edit all of his songs.   How many minutes did it take Elvis to write each song, if each song took the same amount of time to write?

**Expected Ground Truth Answer:** `15`

| Evaluation Mode | Model Predicted Answer | Format Compliant? | SymPy Verified? |
| :--- | :--- | :---: | :---: |
| SYMPY | `(no answer extracted)` | No | FAILED (Incorrect) |

<details>
<summary>View SYMPY Raw Output</summary>

```
Could not parse or solve problem.
```
</details>

---

### Example 3 — [sympy]

**Problem:**
> Expand (4*x - 1)^2

**Expected Ground Truth Answer:** `16*x**2 - 8*x + 1`

| Evaluation Mode | Model Predicted Answer | Format Compliant? | SymPy Verified? |
| :--- | :--- | :---: | :---: |
| SYMPY | `16*x**2 - 8*x + 1` | Yes | PASSED (Correct) |

<details>
<summary>View SYMPY Raw Output</summary>

```
Step 1: Identify Expression to Expand — Start with the unexpanded expression: `(4*x - 1)**2`.
Step 2: Apply Distributive Law / FOIL — Multiply factors and expand all brackets: `16*x**2 - 8*x + 1`.
Final answer: 16*x**2 - 8*x + 1
```
</details>

---

### Example 4 — [huggingface_metamath]

**Problem:**
> If the ratio of $b$ to $a$ is 3, then what is the value of $a$ when $b=12-5a$?

**Expected Ground Truth Answer:** `\frac{3}{2}`

| Evaluation Mode | Model Predicted Answer | Format Compliant? | SymPy Verified? |
| :--- | :--- | :---: | :---: |
| SYMPY | `(no answer extracted)` | No | FAILED (Incorrect) |

<details>
<summary>View SYMPY Raw Output</summary>

```
Could not parse or solve problem.
```
</details>

---

### Example 5 — [huggingface_gsm8k]

**Problem:**
> Monica and Sheila are twins. Their mother gave them $50 and told them to buy some toilet paper and spend the remainder on groceries. The toilet paper cost $12. They bought apples, butter, eggs, and a large ham for twice the cost of the toilet paper. Since they still had some leftover money, they called their mother and she gave them permission to buy whatever they wanted for themselves as long as they shared the money evenly. They saw some boots they really liked, but a pair of boots costs 3 times the amount they had left. How much more would Monica and Sheila each have to add of their own money to buy two pairs of boots?

**Expected Ground Truth Answer:** `35`

| Evaluation Mode | Model Predicted Answer | Format Compliant? | SymPy Verified? |
| :--- | :--- | :---: | :---: |
| SYMPY | `(no answer extracted)` | No | FAILED (Incorrect) |

<details>
<summary>View SYMPY Raw Output</summary>

```
Could not parse or solve problem.
```
</details>

---


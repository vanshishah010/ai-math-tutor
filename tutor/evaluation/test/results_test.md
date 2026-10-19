# AI Math Tutor Evaluation Report

**Total Test Problems Evaluated:** 150

## Overall Benchmark Comparison

| Mode | Correct / Total | Overall Accuracy (95% CI) | Format Compliance (%) | huggingface_gsm8k Acc (95% CI) | huggingface_metamath Acc (95% CI) | sympy Acc (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SYMPY** | 53 / 150 | **35.33%** [28.1%, 43.3%] | 35.33% (53/150) | 0.0% [0.0%, 4.5%] | 0.0% [0.0%, 20.4%] | 100.0% [93.2%, 100.0%] |
| **BASE** | 124 / 150 | **82.67%** [75.8%, 87.9%] | 1.33% (2/150) | 93.9% [86.5%, 97.4%] | 80.0% [54.8%, 93.0%] | 66.0% [52.6%, 77.3%] |
| **FINE-TUNED** | 128 / 150 | **85.33%** [78.8%, 90.1%] | 99.33% (149/150) | 81.7% [72.0%, 88.6%] | 86.7% [62.1%, 96.3%] | 90.6% [79.8%, 95.9%] |

*Note: All confidence intervals are 95% Wilson score intervals.*

## Side-by-Side Qualitative Examples (5 Selected)

### Example 1 — [huggingface_gsm8k]

**Problem:**
> Alex is saving up for a new car. He already has $14,500 saved up and the car costs $14,600. He decides that to make the last of the money he will deliver groceries for people in the neighborhood.  He charges $1.5 per trip and then 5% of the price of the groceries. If he makes 40 trips and has saved up enough, how many dollars' worth of groceries did he deliver?

**Expected Ground Truth Answer:** `800`

| Evaluation Mode | Model Predicted Answer | Format Compliant? | SymPy Verified? |
| :--- | :--- | :---: | :---: |
| SYMPY | `(no answer extracted)` | No | FAILED (Incorrect) |
| BASE | `800` | No | PASSED (Correct) |
| FINE-TUNED | `3200` | Yes | FAILED (Incorrect) |

<details>
<summary>View SYMPY Raw Output</summary>

```
Could not parse or solve problem.
```
</details>

<details>
<summary>View BASE Raw Output</summary>

```
To determine how much worth of groceries Alex delivered, we need to calculate the total amount of money he made from delivering groceries and then use that to find out how much he earned from the 5% commission on the groceries.

First, let's calculate the total amount of money Alex needs to save to buy the car. The car costs $14,600 and Alex already has $14,500 saved up, so he needs:

\[ 14,600 - 14,500 = 100 \text{ dollars} \]

Next, we know that Alex made 40 trips delivering groceries and that he charges $1.5 per trip. So, the total amount of money he made from the trips is:

\[ 40 \times 1.5 = 60 \text{ dollars} \]

Let's denote the total worth of the groceries he delivered as \( G \). According to the problem, he also made 5% of the price of the groceries as commission. Therefore, the commission he made is:

\[ 0.05 \times G \]

The total amount of money Alex made from delivering groceries is the sum of the money from the trips and the commission:

\[ 60 + 0.05G \]

We know that this total amount is equal to the $100 he needed to save. So, we can set up the following equation:

\[ 60 + 0.05G = 100 \]

To find \( G \), we first isolate \( 0.05G \) by subtracting 60 from both sides of the equation:

\[ 0.05G = 40 \]

Next, we solve for \( G \) by dividing both sides by 0.05:

\[ G = \frac{40}{0.05} = 800 \]

Therefore, the total worth of the groceries Alex delivered is:

\[ \boxed{800} \]
```
</details>

<details>
<summary>View FINE-TUNED Raw Output</summary>

```
Step 1: He needs $14,600-$14,500=$100 more
Step 2: He makes $1.5 per trip so he makes $1.5*40=$60 from the trips
Step 3: He makes $60+$100=$160 from the groceries
Step 4: He makes 5% of the price of the groceries so he made $160=$160/5=$3200 worth of groceries
Final answer: 3200
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
| BASE | `15` | No | PASSED (Correct) |
| FINE-TUNED | `15` | Yes | PASSED (Correct) |

<details>
<summary>View SYMPY Raw Output</summary>

```
Could not parse or solve problem.
```
</details>

<details>
<summary>View BASE Raw Output</summary>

```
To solve this problem, we need to determine how much time Elvis spent writing each song. Here's a step-by-step approach:

1. Convert all time units to minutes for consistency.
2. Calculate the total time spent recording all songs.
3. Calculate the total time spent editing all songs.
4. Subtract the total recording and editing time from the total time spent in the studio to find the time spent writing.
5. Divide the time spent writing by the number of songs to find the time spent writing each song.

Let's break this down and implement it in Python.
```python
# Total time spent in the studio in minutes
total_time_studio = 5 * 60  # 5 hours converted to minutes

# Time spent recording each song in minutes
time_per_song_record = 12

# Total number of songs
num_songs = 10

# Total time spent recording all songs
total_time_record = time_per_song_record * num_songs

# Total time spent editing all songs
total_time_edit = 30

# Total time spent writing all songs
total_time_writing = total_time_studio - (total_time_record + total_time_edit)

# Time spent writing each song
time_per_song_writing = total_time_writing / num_songs

print(time_per_song_writing)
```
```output
15.0
```
Elvis spent \(\boxed{15}\) minutes writing each song.
```
</details>

<details>
<summary>View FINE-TUNED Raw Output</summary>

```
Step 1: He spent 5 hours in the studio, which is 5*60=300 minutes.
Step 2: He spent 30 minutes editing all of his songs, so he spent 300-30=270 minutes writing and recording his songs.
Step 3: He spent 12 minutes recording each song, so he spent 12*10=120 minutes recording all of his songs.
Step 4: So he spent 270-120=150 minutes writing his songs.
Step 5: He spent 150/10=15 minutes writing each song.
Final answer: 15
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
| BASE | `16x^2 - 8x + 1` | No | PASSED (Correct) |
| FINE-TUNED | `16*x**2 - 8*x + 1` | Yes | PASSED (Correct) |

<details>
<summary>View SYMPY Raw Output</summary>

```
Step 1: Identify Expression to Expand — Start with the unexpanded expression: `(4*x - 1)**2`.
Step 2: Apply Distributive Law / FOIL — Multiply factors and expand all brackets: `16*x**2 - 8*x + 1`.
Final answer: 16*x**2 - 8*x + 1
```
</details>

<details>
<summary>View BASE Raw Output</summary>

```
To expand the expression \((4x - 1)^2\), we can use the formula for the square of a binomial, which is \((a - b)^2 = a^2 - 2ab + b^2\). Here, \(a = 4x\) and \(b = 1\).

Let's apply the formula step by step:

1. Identify \(a\) and \(b\):
   \(a = 4x\)
   \(b = 1\)

2. Substitute \(a\) and \(b\) into the formula \((a - b)^2 = a^2 - 2ab + b^2\):
   \((4x - 1)^2 = (4x)^2 - 2(4x)(1) + (1)^2\)

3. Calculate each term:
   \((4x)^2 = 16x^2\)
   \(-2(4x)(1) = -8x\)
   \((1)^2 = 1\)

4. Combine the terms:
   \((4x - 1)^2 = 16x^2 - 8x + 1\)

So, the expanded form of \((4x - 1)^2\) is \(\boxed{16x^2 - 8x + 1}\).
```
</details>

<details>
<summary>View FINE-TUNED Raw Output</summary>

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
| BASE | `\frac{3}{2}` | No | PASSED (Correct) |
| FINE-TUNED | `1.5` | Yes | PASSED (Correct) |

<details>
<summary>View SYMPY Raw Output</summary>

```
Could not parse or solve problem.
```
</details>

<details>
<summary>View BASE Raw Output</summary>

```
To solve the problem, we start by using the given information that the ratio of \( b \) to \( a \) is 3. This can be written as:

\[
\frac{b}{a} = 3
\]

This implies:

\[
b = 3a
\]

We are also given the equation:

\[
b = 12 - 5a
\]

Since both expressions are equal to \( b \), we can set them equal to each other:

\[
3a = 12 - 5a
\]

Next, we solve for \( a \) by isolating \( a \) on one side of the equation. First, add \( 5a \) to both sides:

\[
3a + 5a = 12
\]

This simplifies to:

\[
8a = 12
\]

Now, divide both sides by 8:

\[
a = \frac{12}{8} = \frac{3}{2}
\]

Thus, the value of \( a \) is:

\[
\boxed{\frac{3}{2}}
\]
```
</details>

<details>
<summary>View FINE-TUNED Raw Output</summary>

```
Step 1: We are given that the ratio of $b$ to $a$ is 3, so we can write $\frac{b}{a} = 3$.
Step 2: We are also given that $b = 12 - 5a$.
Step 3: Substituting the value of $b$ from Step 2 into the ratio equation, we get $\frac{12 - 5a}{a} = 3$.
Step 4: Multiplying both sides by $a$, we get $12 - 5a = 3a$.
Step 5: Combining like terms, we get $12 = 8a$.
Step 6: Dividing both sides by 8, we get $a = \boxed{1.5}$.
Final answer: 1.5
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
| BASE | `35` | No | PASSED (Correct) |
| FINE-TUNED | `21` | Yes | FAILED (Incorrect) |

<details>
<summary>View SYMPY Raw Output</summary>

```
Could not parse or solve problem.
```
</details>

<details>
<summary>View BASE Raw Output</summary>

```
Let's start by calculating the total amount of money Monica and Sheila have after buying the toilet paper and groceries.

1. The cost of the toilet paper is $12.
2. The cost of the groceries is twice the cost of the toilet paper, which is \(2 \times 12 = 24\) dollars.
3. The total amount spent on the toilet paper and groceries is \(12 + 24 = 36\) dollars.
4. The remaining money is \(50 - 36 = 14\) dollars.

Next, we need to determine the cost of the boots. The cost of a pair of boots is 3 times the amount of money they have left, which is \(3 \times 14 = 42\) dollars. Since they need to buy two pairs of boots, the total cost for the boots is \(2 \times 42 = 84\) dollars.

Since they only have $14 left, they need to add \(84 - 14 = 70\) dollars to buy the boots. Since they need to share the cost evenly, each of them needs to add \(70 \div 2 = 35\) dollars.

Therefore, the amount each twin needs to add is \(\boxed{35}\).
```
</details>

<details>
<summary>View FINE-TUNED Raw Output</summary>

```
Step 1: The apples, butter, eggs, and ham cost 2 * $12 = $24.
Step 2: They spent $12 + $24 = $36 on the toilet paper and groceries.
Step 3: They had $50 - $36 = $14 left.
Step 4: A pair of boots costs 3 * $14 = $42.
Step 5: They need $42 * 2 = $84 to buy two pairs of boots.
Step 6: Monica and Sheila each need to add $84 / 2 = $42 / 2 = $21 of their own money to buy two pairs of boots.
Final answer: 21
```
</details>

---


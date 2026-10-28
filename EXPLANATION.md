# The SHMS & AERIS Guide
### Everything about this project and the AI inside it, explained from zero

This guide assumes you know **nothing** about Artificial Intelligence. Every idea is explained in everyday
language, with small examples you can work out on paper. By the end you should be able to explain every
part of this project in your own words, which is exactly what a viva needs.

> **How to read this:** go in order. Each chapter builds on the one before it. Do the small examples
> yourself with a pen; that is how the ideas stick.

---

## Table of contents

1. [The story: what problem are we solving?](#1-the-story-what-problem-are-we-solving)
2. [Spacecraft, sensors and telemetry](#2-spacecraft-sensors-and-telemetry)
3. [What is AI, Machine Learning and Deep Learning?](#3-what-is-ai-machine-learning-and-deep-learning)
4. [Data: the food of AI](#4-data-the-food-of-ai)
5. [How machines learn: the main types](#5-how-machines-learn-the-main-types)
6. [Stage 1 – The Forecaster (Linear Regression)](#6-stage-1--the-forecaster-linear-regression)
7. [Features: giving the model clues](#7-features-giving-the-model-clues)
8. [Stage 2 – The Anomaly Detective (Decision Trees and Random Forest)](#8-stage-2--the-anomaly-detective-decision-trees-and-random-forest)
9. [Logistic Regression – the model that lost](#9-logistic-regression--the-model-that-lost)
10. [Stage 3 – Sensor Mini Brains (Neural Networks)](#10-stage-3--sensor-mini-brains-neural-networks)
11. [Training and testing fairly](#11-training-and-testing-fairly)
12. [From probabilities to alarms](#12-from-probabilities-to-alarms)
13. [Evaluation metrics: is the model any good?](#13-evaluation-metrics-is-the-model-any-good)
14. [AERIS's real results and what they mean](#14-aeriss-real-results-and-what-they-mean)
15. [The AI Project Cycle and Model Lifecycle](#15-the-ai-project-cycle-and-model-lifecycle)
16. [Generative AI: how Gemini writes the report](#16-generative-ai-how-gemini-writes-the-report)
17. [The website: what happens when you press Analyze](#17-the-website-what-happens-when-you-press-analyze)
18. [Storytelling with data: reading the charts](#18-storytelling-with-data-reading-the-charts)
19. [AI ethics in this project](#19-ai-ethics-in-this-project)
20. [Tools and technologies](#20-tools-and-technologies)
21. [Glossary](#21-glossary)
22. [Viva questions and answers](#22-viva-questions-and-answers)
23. [Syllabus map](#23-syllabus-map)

---

## 1. The story: what problem are we solving?

Imagine you are a doctor responsible for **one patient**. You can check their temperature, pulse and blood
pressure yourself. Easy.

Now imagine you are responsible for **one patient who sends you 50 readings every second, all day, for
years**. You would need to read millions of numbers. You would get tired, you would miss things, and a
small warning sign could turn into an emergency.

That is the problem with spacecraft. A satellite or a Mars rover has many sensors, and each one sends
readings back to Earth non-stop. Engineers cannot look at every number. If a battery starts to overheat or
a wheel starts to wobble, the early signs are hidden inside that huge pile of data.

**Our solution:** a website called **SHMS (Spacecraft Health Monitoring System)**. You upload a file of
sensor readings, press one button, and an AI model called **AERIS** reads every single number and tells
you in plain words:

- **HEALTHY** – nothing unusual found.
- **WARNING** – something looks odd; an engineer should check.
- **DANGER** – strong or long-lasting problems; act quickly.

It also gives a **health score from 0 to 100**, easy charts, a report written in simple language by
Google's Gemini AI, and scores that show how much you can trust the answer.

**AERIS** stands for **A**erospace **E**quipment & **R**eliability **I**ntelligence **S**ystem.

---

## 2. Spacecraft, sensors and telemetry

### What is a sensor?
A sensor is a device that **measures something** and turns it into a number.

| Everyday sensor | What it measures |
|---|---|
| Thermometer | Temperature |
| Phone battery indicator | Battery charge |
| Car speedometer | Speed |
| Fitness band | Heart rate |

A spacecraft has the same kind of sensors: temperatures of parts, battery voltage, power use, wheel speed,
pressure in fuel tanks, and many more.

### What is telemetry?
**Tele** means "far away" and **metry** means "measuring". Telemetry is **measuring something from far away**.

When the Curiosity rover on Mars reads its battery voltage, it sends that number across space to Earth.
That stream of numbers is telemetry. Think of it as the spacecraft's **health report, sent every few seconds**.

### What does telemetry look like?
Just a long list of numbers in time order:

```
Time:     1     2     3     4     5     6     7     8
Reading: 0.50  0.51  0.50  0.52  0.51  0.95  0.96  0.52
```

Look at readings 6 and 7. They suddenly jumped from about 0.5 to about 0.95, then came back. That jump
might be nothing, or it might be the first sign of a problem. That "unusual" behaviour is called an
**anomaly**.

### What is an anomaly?
An anomaly is **anything that does not look like normal behaviour**. Examples:

- **Spike:** a sudden jump up or down (like readings 6–7 above).
- **Level shift:** the values move to a new level and stay there (like a fever that doesn't go away).
- **Frozen sensor:** the readings stop changing at all (a broken sensor often does this).
- **Strange pattern:** the shape of the signal changes, even if values stay in range.

Finding these is called **anomaly detection**, and that is AERIS's main job.

### What is a "channel"?
In our NASA dataset, each sensor is called a **channel**, with names like `A-1`, `E-1`, `P-11`, `T-1`.
Each channel is one sensor. NASA kept the real sensor names secret, so we don't know exactly what each one
measures. The letter only tells you which group of sensors a channel belongs to; it does not mean
"T = temperature".

---

## 3. What is AI, Machine Learning and Deep Learning?

### Artificial Intelligence (AI)
AI means **making computers do things that normally need human intelligence**: recognising faces,
understanding language, making decisions, spotting problems.

### The old way vs. the AI way
Suppose we want a computer to find problems in a temperature sensor.

**Old way (rules written by a person):**
```
IF temperature > 80 THEN raise alarm
```
This is simple, but weak. What if 80 is normal during a hot part of the orbit? What if the problem is
the sensor freezing at 40? A person would need to write thousands of rules, and still miss things.

**AI way (Machine Learning):** instead of writing the rules ourselves, we **show the computer lots of
examples** of normal and abnormal data, and it **figures out the rules by itself**.

### Machine Learning (ML)
Machine Learning is the part of AI where a computer **learns patterns from data** instead of being told
exact rules.

**Example:** how did you learn what a dog looks like? Nobody gave you a rule like "4 legs + tail + fur =
dog" (a cat has those too!). You saw many dogs, and your brain learned the pattern. ML works the same way.

### Deep Learning (DL)
Deep Learning is a type of Machine Learning that uses **neural networks** (computer "brains" made of
layers of connected units). We use small neural networks in Stage 3 of AERIS.

```
┌─────────────── Artificial Intelligence ───────────────┐
│   ┌────────────── Machine Learning ──────────────┐    │
│   │   ┌──────── Deep Learning ────────┐          │    │
│   │   │   (neural networks)           │          │    │
│   │   └───────────────────────────────┘          │    │
│   └──────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────┘
```

### What is a "model"?
A **model** is what the computer **builds after learning**. It is like a recipe the computer wrote for
itself. You give the model new data, and it gives you an answer (a **prediction**).

- **Training** = the learning phase (making the recipe).
- **Prediction / Inference** = using the recipe on new data.

**AERIS is our model.** It was trained once (by running `python aeris.py`) and saved in the `models/`
folder. The website just loads it and uses it.

---

## 4. Data: the food of AI

An AI model is only as good as the data it learns from. Bad data → bad model. This is often said as
**"garbage in, garbage out"**.

### Our dataset
We use the **NASA SMAP & MSL Anomaly Detection Dataset** (downloaded from Kaggle).

- **SMAP** = Soil Moisture Active Passive, a NASA satellite that measures moisture in soil from space.
- **MSL** = Mars Science Laboratory, better known as the **Curiosity rover** on Mars.
- **82 channels** (sensors) in total.
- **105 anomalies** marked by real NASA engineers.

### The folders
```
Dataset/
├── data/train/   ← normal recordings of each sensor (one .npy file per sensor)
├── data/test/    ← recordings that contain real problems
└── labeled_anomalies.csv   ← NASA's answer key: where the problems are
```

- **Train files:** a stretch of time where the sensor behaved normally. Good for learning "what normal looks like".
- **Test files:** a later stretch of time that contains real anomalies.
- **labeled_anomalies.csv:** the answer key. For example, it says *"in channel E-1, readings 5000–5030 and
  5610–6086 are anomalies"*. These answers are called **labels**.

### What is a .npy file?
`.npy` is a file format from **NumPy** (a Python library for numbers). It stores a big table of numbers
very efficiently. In our files:

- **Column 0** = the actual sensor reading (the telemetry).
- **The other columns** (24 or 54 of them) = 0/1 flags showing which commands were sent to the spacecraft at
  that moment. AERIS focuses on column 0.

### Why convert everything to CSV?
**CSV** (Comma-Separated Values) is the simplest table format in the world, like a spreadsheet saved as text:

```
time,temperature,pressure
1,20.1,101.3
2,20.3,101.2
3,20.2,101.4
```

Our website accepts `.npy`, `.csv`, `.xlsx`, `.xls`, `.json` and `.txt` files, but it **converts every file to
CSV first**, then analyses the CSV. That way the rest of the system only has to understand one format, and
**Pandas** (a Python library) reads CSV files very easily. You can even download the converted CSV from
the results page.

### Data cleaning
Real data can have holes (missing values) or broken values. Before analysis, AERIS:

- Fills missing values using nearby values (called **interpolation**: if reading 5 is missing and readings
  4 and 6 are 10 and 12, it fills in 11).
- Skips columns that are not real sensors, such as time columns, index counters and on/off flags.

### Scaling (normalisation)
Different sensors use different units. One might range from 0 to 5 volts, another from −40 to 120 °C. To
compare them fairly, AERIS **scales** every signal to the range **−1 to 1**:

```
scaled = 2 × (value − smallest) ÷ (largest − smallest) − 1
```

**Example:** a temperature signal goes from 20 to 30.
- 20 → 2 × 0/10 − 1 = **−1**
- 25 → 2 × 5/10 − 1 = **0**
- 30 → 2 × 10/10 − 1 = **1**

### Data augmentation (making extra practice data)
NASA's signals mostly look like flat lines with steps. When we first tested AERIS on a smooth wave (like a
temperature that rises and falls during each orbit), it wrongly thought the wave was a problem, because it
had never seen one.

**Fix:** we created **72 practice signals** in code: smooth waves, noisy signals, slow drifts and on/off
steps. Half are healthy, and half have a fault added on purpose (a spike, a jump or a frozen section), with
the correct answer attached. This is called **data augmentation**: creating extra realistic training
examples so the model learns more variety.

**Everyday example:** to help a child recognise cats, you show photos of cats in many colours, sizes and
poses, not just one orange cat.

---

## 5. How machines learn: the main types

### Supervised learning: learning with answers
You give the model examples **and the correct answers**.

**Example:** you show 1000 emails, each marked "spam" or "not spam". The model learns what spam looks like.

In AERIS: the **Random Forest** learns from NASA's answer key (which readings were anomalies). This is
supervised learning.

### Unsupervised learning: learning without answers
You give the model data **without answers**, and it finds patterns by itself.

**Example:** you give a pile of photos with no labels, and the model groups similar photos together.

In AERIS: the **mini brains** learn only from normal recordings. Nobody tells them where the problems are;
they just learn "this is what normal looks like". Anything very different surprises them.

### Regression vs Classification
Both are supervised learning, but they predict different kinds of answers.

| | Regression | Classification |
|---|---|---|
| Predicts | A **number** | A **category** |
| Example | Tomorrow's temperature: 31.4 °C | Will it rain? Yes / No |
| In AERIS | Next sensor reading (Stage 1, Stage 3) | Normal or anomaly (Stage 2) |

---

## 6. Stage 1 – The Forecaster (Linear Regression)

### The idea
If you know a sensor's last few readings, you can usually guess the next one. If the real next reading is
**very different** from your guess, something surprising happened, and surprises are often anomalies.

**Everyday example:** your friend arrives at school at 8:00 every day: 7:58, 8:01, 7:59, 8:00… You guess
tomorrow they'll come at about 8:00. If they arrive at 11:30, that's a surprise, and maybe something is wrong.

### What is Linear Regression?
Linear Regression finds the **best straight-line relationship** between inputs and an output.

The simplest version has one input:
```
output = m × input + c
```
- **m** = slope (how much the output changes when the input goes up by 1)
- **c** = intercept (the starting value)

**Example:** a phone charges 2% every minute, starting at 10%.
```
battery = 2 × minutes + 10
After 20 minutes: 2 × 20 + 10 = 50%
```
Here the model "learned" m = 2 and c = 10.

### How does it "learn" m and c?
It tries values that make the **errors as small as possible**. For every example, it measures
(real value − predicted value), squares it (so negatives don't cancel positives), and adds them all up. The
line with the **smallest total squared error** wins. This is called the **least squares** method.

### How AERIS uses it
AERIS's forecaster looks at the **last 30 readings** and predicts the next one:
```
next reading = w1 × reading(t−1) + w2 × reading(t−2) + … + w30 × reading(t−30) + c
```
It learned the 30 weights (w1…w30) from about **194,000 normal readings** from all NASA sensors.

### The "surprise"
For every reading:
```
forecast error (surprise) = | real reading − predicted reading |
```
(the bars | | mean "ignore the minus sign").

| Time | Real | Predicted | Surprise |
|---|---|---|---|
| 1 | 0.50 | 0.50 | 0.00 |
| 2 | 0.51 | 0.50 | 0.01 |
| 3 | 0.95 | 0.51 | **0.44** ← big surprise! |

The surprise is not the final answer by itself. It becomes one of the **clues** for Stage 2.

The quality of the forecaster is measured with **MSE, RMSE, MAE and MAPE** (see Chapter 13).

---

## 7. Features: giving the model clues

### What is a feature?
A **feature** is one piece of information (a clue) given to a model to help it decide.

**Example:** to guess if a fruit is a mango or a banana, useful features are colour, shape, length and
weight. "The day you bought it" is a useless feature.

Choosing and creating good features is called **feature engineering**, and it is one of the most important
skills in Machine Learning.

### AERIS's 27 clues
For **every single reading**, AERIS calculates 27 clues. Here are the most important ones, explained:

**1. Rolling mean (moving average)**
The average of the last 50 readings.
```
Readings: 4, 6, 5, 5  →  rolling mean = (4+6+5+5) ÷ 4 = 5
```
It shows the "normal level" right now.

**2. Standard deviation (spread)**
How much the readings jump around their average. Small = calm and steady. Large = jumpy.
```
Calm:  5, 5, 5, 5   → standard deviation 0
Jumpy: 1, 9, 2, 8   → standard deviation about 3.5
```

**3. Z-score (how unusual is this reading?)**
```
z = (reading − average) ÷ standard deviation
```
**Example:** class test marks average 60 with standard deviation 10. You scored 90.
z = (90 − 60) ÷ 10 = **3** → you are 3 "steps" above average, which is very unusual. In AERIS a large
z-score means the reading is far from its recent normal.

**4. Rarity**
How rarely this value appears in the file. If 99% of readings are around 0.5 and one reading is 0.95, that
value is very rare, and rare is suspicious.

**5. Forecast error (surprise)**
From Stage 1: how far the reading was from the forecaster's guess.

**6. Comparison with the sensor's normal recording**
NASA gives a normal recording for each sensor. AERIS checks: *has this sensor ever shown this value during
normal operation?* If a value was **never seen** in the normal recording, that is a strong warning.
(For files that aren't from NASA, the first 30% of the file is used as the "normal" reference.)

**7. Long-term change**
The difference between the reading and the average of the last 250 or 1000 readings. This catches slow
drifts that short windows miss.

**8. Change from the previous reading**
A sudden jump between two readings in a row.

Together these 27 numbers describe each reading from many angles, like a detective's notebook.

---

## 8. Stage 2 – The Anomaly Detective (Decision Trees and Random Forest)

### A Decision Tree
A decision tree is a **flowchart of yes/no questions** that leads to an answer.

**Everyday example – should I take an umbrella?**
```
              Is the sky cloudy?
              /                \
           Yes                  No
            |                    |
   Is humidity > 80%?        No umbrella
     /          \
   Yes           No
    |             |
 Umbrella    No umbrella
```

A decision tree in AERIS might look like:
```
             Is the surprise > 0.3?
             /                 \
          Yes                   No
           |                     |
  Is the value rare?        Is the z-score > 4?
     /       \                /         \
   Yes        No            Yes          No
    |          |             |            |
 ANOMALY    normal        ANOMALY       normal
```
**The computer builds these questions itself during training.** It picks the questions that best separate
anomalies from normal readings in the training data.

**Problem with one tree:** it can become too specific to the training data (it memorises instead of
learning) and make silly mistakes on new data.

### A Random Forest
A Random Forest is **many decision trees that vote**.

**Everyday example:** instead of asking one friend whether a movie is good, you ask 120 friends who each
watched different parts of it. The majority answer is much more reliable than any single friend.

How it works:
1. Build many trees (AERIS uses **120 trees**).
2. Each tree learns from a **random sample** of the data and looks at a **random subset of the clues**. That
   is why it's called "random"; it makes the trees different from each other.
3. For a new reading, every tree votes "normal" or "anomaly".
4. The share of "anomaly" votes becomes the **probability**.
   If 90 of 120 trees say anomaly → probability = 90 ÷ 120 = **0.75**.

### Why did we choose Random Forest?
We tested three models fairly (with cross validation, Chapter 11) and compared their F1 scores
(Chapter 13):

| Model | F1 score |
|---|---|
| Logistic Regression | 0.389 |
| Decision Tree | 0.374 |
| **Random Forest** | **0.482** ← winner |

This is called **model selection**: try several models and pick the best one using a fair test.

### Class imbalance
Only about **12.5%** of readings are anomalies; the other 87.5% are normal. If we're not careful, a model
can get lazy and say "normal" all the time. We told the Random Forest to give **more importance to the rare
anomaly examples** (a setting called `class_weight="balanced"`), so it pays attention to them.

---

## 9. Logistic Regression – the model that lost

Even though it lost, you should understand it, because it's a classic classification model.

Despite its name, Logistic Regression is used for **classification** (yes/no). It works like Linear
Regression, but it squeezes the output through an S-shaped curve (the **sigmoid** function) so the answer
is always between 0 and 1, which we read as a probability.

```
probability
   1 |            ______
     |          /
 0.5 |        /
     |      /
   0 |_____/
      ---------------------> score
```

**Example:** predicting whether a student passes from hours studied. 0 hours → probability 0.05,
5 hours → 0.5, 10 hours → 0.95.

It lost in our project because anomalies follow complicated, non-straight-line patterns, which tree-based
models handle better.

---

## 10. Stage 3 – Sensor Mini Brains (Neural Networks)

### The problem we solved
One general model has to understand all 81 sensors at once. But every sensor has its own "personality": one
is always flat, another wiggles a lot. A wiggle that's normal for one sensor can be a problem for another.

**Solution:** give **every NASA sensor its own small brain** that learns only that sensor's normal
behaviour. Like a family doctor who knows *your* normal blood pressure, not just the average person's.

### What is a neural network?
A neural network is a model inspired (loosely) by the human brain. It's made of **neurons** arranged in
**layers**:

```
 Input layer          Hidden layers          Output layer
 (last 60 readings)   (64 neurons, 32 neurons)  (next reading)

   o ─┐
   o ─┼──►  o o o o …  ──►  o o o …  ──►   o
   o ─┤     (64)            (32)
   …  ┘
```

**What does one neuron do?** It takes some numbers in, multiplies each by a **weight** (importance), adds
them up, adds a **bias**, and passes the result through an **activation function** (a simple rule such as
"if the result is negative, output 0").

**Example neuron:**
```
inputs:  3 and 5
weights: 0.5 and 0.2
bias:    1
sum = 3×0.5 + 5×0.2 + 1 = 1.5 + 1.0 + 1 = 3.5
activation (ReLU: keep positive numbers, turn negatives into 0) → output 3.5
```
Many neurons connected in layers can learn very complicated patterns.

**How does it learn?** It makes a guess, measures how wrong it was (the error), and then slightly adjusts
all the weights to make the error smaller. It repeats this thousands of times. This adjusting process is
called **backpropagation** with **gradient descent**, which you can picture as walking downhill step by step
until you reach the lowest point (the smallest error).

The type we use is called an **MLP (Multi-Layer Perceptron)**, the most basic kind of neural network.

### How the mini brains work in AERIS
1. Each of the 81 NASA sensors gets its own neural network (60 inputs → 64 → 32 → 1 output).
2. It learns **only from that sensor's normal recording**, predicting the next reading from the last 60.
   It never sees the answer key, so this part is **unsupervised** in spirit.
3. It also remembers how big its own errors normally are.
4. On new data, it measures its **surprise** in units of its normal error. A surprise of 10 means "10 times
   more surprised than usual".
5. This surprise is given to the Random Forest as **3 extra clues**, making 30 clues in total. The Random
   Forest that uses them is called the **expert model**.

### Expert mode vs general mode
- **Expert mode:** used when AERIS recognises the file as a NASA test file, **even if it was renamed**
  (it compares the actual numbers with NASA's files). It uses that sensor's mini brain.
- **General mode:** used for any other file (your own CSV or Excel). It uses the general Random Forest
  without a mini brain, because there is no brain for an unknown sensor.

This combination improved AERIS's F1 score on NASA sensors from **0.315 to 0.542**.

---

## 11. Training and testing fairly

### Why test at all?
A model that does well on the data it learned from might simply have **memorised** it.

**Example:** a student memorises the answers to last year's paper. They score 100% on that paper, but fail
the real exam with new questions. We want to know how AERIS does on **new questions**.

### Overfitting and underfitting
- **Overfitting:** the model memorises the training data, including its random noise, and does badly on
  new data. (The student who memorised answers.)
- **Underfitting:** the model is too simple to learn even the training data. (The student who didn't study.)
- **Good fit:** learns the real patterns and works on new data.

**How AERIS avoids overfitting:**
- Testing on sensors it has never seen (train-test split).
- Cross validation.
- Limiting tree depth (each tree can ask at most 14 questions in a row) and requiring at least 20 examples
  per final decision (so trees can't memorise single odd readings).
- Data augmentation (more variety).
- Early stopping for the neural networks (they stop training when they stop improving on a check set).

### Train-test split
We split the data into two parts:
- **Training set:** the model learns from this.
- **Test set:** kept hidden and used only at the end, for the final exam.

**Important detail:** we split **by sensor**, not by reading. 64 sensors were for training and **17 sensors
were completely hidden**. If we had split readings randomly, readings from the same sensor would be in both
sets, and the test would be too easy (like seeing half the exam answers in advance).

### Cross validation (K-Fold)
Testing only once can be lucky or unlucky. Cross validation tests several times.

**5-fold cross validation:**
1. Split the sensors into 5 equal groups (folds).
2. Train on 4 groups and test on the 1 left out.
3. Repeat 5 times, so each group is the test group once.
4. Average the 5 scores.

```
Round 1:  [TEST] [train] [train] [train] [train]
Round 2:  [train] [TEST] [train] [train] [train]
Round 3:  [train] [train] [TEST] [train] [train]
Round 4:  [train] [train] [train] [TEST] [train]
Round 5:  [train] [train] [train] [train] [TEST]
```

We used this to choose the best model (Random Forest) and the best alarm settings.

### Cross-fitting (the expert model's trick for honest scores)
For the expert model we trained **5 copies**, each one without one fifth of the sensors. When you analyse,
say, sensor E-1, AERIS uses the copy that **never saw E-1's answers**. So the score shown on the results page
is a genuine test, not a memorised one. This is how we keep the evaluation honest for **every** NASA sensor.

### Hyperparameters
**Parameters** are learned by the model (like the weights in Linear Regression).
**Hyperparameters** are settings **we choose before training**.

**Example:** when baking a cake, the recipe's ingredients are like data, the oven temperature you choose is a
hyperparameter, and how the cake turns out is the model.

AERIS's hyperparameters (found in `CONFIG` in `aeris.py`):

| Hyperparameter | Value | Meaning |
|---|---|---|
| lags | 30 | Readings the forecaster looks back at |
| short_window | 50 | Window for rolling mean / standard deviation |
| long_window | 250 | Window for long-term trends |
| n_estimators | 120 | Number of trees in the Random Forest |
| max_depth | 14 | Maximum questions in a row per tree |
| holdout_fraction | 0.2 | 20% of sensors hidden for testing |
| cv_folds | 5 | Folds in cross validation |
| threshold | 0.4 | Risk level that triggers an alarm (tuned) |
| smoothing | 75 | How many readings the risk score is averaged over (tuned) |
| brain_lags | 60 | Readings each mini brain looks at |

---

## 12. From probabilities to alarms

The Random Forest gives a **probability** (risk score) between 0 and 1 for every reading. How do we turn
that into a decision?

**Step 1 – Smoothing.** Single readings can be noisy, so we average the risk over 75 neighbouring readings.
One random blip won't trigger an alarm, but a steady rise will.

**Step 2 – Threshold.** If the smoothed risk is **0.4 or more**, the reading is marked as an anomaly.
This 0.4 was chosen during cross validation because it gave the best F1 score.

**Step 3 – Events.** Neighbouring anomaly readings are grouped into one **event** (one period of unusual
behaviour). Events closer than 20 readings are merged, and tiny events shorter than 5 readings are ignored.

**Step 4 – Status.**
- No events → **HEALTHY**
- 10% or more of readings are anomalies, **or** the strongest risk is 0.9 or more → **DANGER**
- Otherwise → **WARNING**

**Step 5 – Health score (0 to 100).** A simple, explainable formula:
```
health = 100 − (percentage of anomalous readings × 3) − (strongest risk × 30, if any event)
```
**Example:** 5% of readings are anomalous and the strongest risk is 0.8:
```
health = 100 − (5 × 3) − (0.8 × 30) = 100 − 15 − 24 = 61
```
If a file has several sensors, the overall health is the **lowest** sensor's score, because a spacecraft is
only as healthy as its weakest part.

---

## 13. Evaluation metrics: is the model any good?

We need numbers to judge the model. There are two families: metrics for **classification** (Stage 2) and
metrics for **regression** (Stage 1).

### The confusion matrix
Every prediction falls into one of four boxes:

| | AERIS said **anomaly** | AERIS said **normal** |
|---|---|---|
| **Really an anomaly** | ✅ True Positive (TP) | ❌ False Negative (FN), *missed it* |
| **Really normal** | ❌ False Positive (FP), *false alarm* | ✅ True Negative (TN) |

**Memory trick:** "Positive/Negative" is what the **model said**. "True/False" is whether it was **right**.

### Worked example
A sensor has **100 readings**; **10** are real anomalies. AERIS flags **8** readings, and **6** of those are
real anomalies.

- TP = 6 (correctly flagged)
- FP = 8 − 6 = 2 (false alarms)
- FN = 10 − 6 = 4 (missed)
- TN = 100 − 6 − 2 − 4 = 88 (correctly left alone)

### Accuracy
```
Accuracy = (TP + TN) ÷ total = (6 + 88) ÷ 100 = 0.94 = 94%
```
Sounds amazing, right? Be careful:

> **The accuracy trap:** a lazy model that **always** says "normal" would get TN = 90 and FN = 10, so
> accuracy = 90 ÷ 100 = **90%**, while finding **zero** problems! When anomalies are rare, accuracy
> is misleading. That's why we need the next metrics.

### Precision: "When AERIS raises an alarm, how often is it right?"
```
Precision = TP ÷ (TP + FP) = 6 ÷ (6 + 2) = 0.75
```
75% of alarms were real. Low precision means many false alarms, which annoys engineers ("the boy who cried
wolf").

### Recall: "Out of all real problems, how many did AERIS find?"
```
Recall = TP ÷ (TP + FN) = 6 ÷ (6 + 4) = 0.60
```
It found 60% of the anomalous readings. Low recall means problems are being missed, which is dangerous
for a spacecraft.

### F1 score: one number that balances both
```
F1 = 2 × (Precision × Recall) ÷ (Precision + Recall)
   = 2 × (0.75 × 0.60) ÷ (0.75 + 0.60)
   = 0.90 ÷ 1.35 = 0.667
```
F1 is high only when **both** precision and recall are high. 1.0 is perfect. It is our main score.

### The precision–recall trade-off
- Lower the alarm threshold → more alarms → **recall goes up**, but **precision goes down** (more false alarms).
- Raise the threshold → fewer alarms → **precision goes up**, but **recall goes down** (more misses).

That's why we tuned the threshold to get the best F1, a balance of both.

### Event-level metrics (shown as tiles on the results page)
Readings are strict: if NASA's anomaly window is readings 5000–5100 and AERIS flags 5010–5090, the few
edge readings count as "missed" even though AERIS clearly found the problem. So we also count **events**:

- **Anomaly events caught:** how many real problems got at least one alarm.
- **Alarms that were real:** how many of AERIS's alarms overlapped a real problem.

### Regression metrics (for the forecaster)
Suppose the forecaster guessed **10, 12, 14** and the real readings were **11, 12, 16**.

Errors (real − guess): **1, 0, 2**

**MAE – Mean Absolute Error** (average size of the errors)
```
MAE = (1 + 0 + 2) ÷ 3 = 1.0
```

**MSE – Mean Squared Error** (average of squared errors)
```
MSE = (1² + 0² + 2²) ÷ 3 = (1 + 0 + 4) ÷ 3 = 1.67
```
Squaring punishes big mistakes more: an error of 2 counts 4 times as much as an error of 1.

**RMSE – Root Mean Squared Error** (square root of MSE, back in the original units)
```
RMSE = √1.67 = 1.29
```
Read it as "the forecaster is typically off by about 1.29".

**MAPE – Mean Absolute Percentage Error** (error as a percentage of the real value)
```
errors as %:  1/11 = 9.1%,  0/12 = 0%,  2/16 = 12.5%
MAPE = (9.1 + 0 + 12.5) ÷ 3 = 7.2%
```
(Our scaled readings can be 0, and dividing by 0 is impossible, so AERIS shifts the values up by 2 before
computing MAPE.)

**For all four: smaller is better.**

---

## 14. AERIS's real results and what they mean

### Expert mode (NASA sensors)
Tested on **all 81 labelled sensors**, each judged by a model that never saw its answers:

| Metric | Score | Plain meaning |
|---|---|---|
| Events caught | **97 of 105 (92%)** | Finds almost every real problem |
| Accuracy | 0.88 | 88% of readings labelled correctly |
| Precision | 0.52 | About half of the alarms are real problems |
| Recall | 0.57 | Finds 57% of the individual anomalous readings |
| F1 score | 0.54 | Balanced score |

### General mode (any other file)
Tested on **17 hidden NASA sensors** it never saw:

| Metric | Score |
|---|---|
| Events caught | 24 of 26 (92%) |
| Precision | 0.24 |
| Recall | 0.47 |
| F1 score | 0.32 |
| Forecaster RMSE | 0.25 (on a −1 to 1 scale) |

### How to explain these honestly in the viva
- **Strength:** AERIS catches about **9 out of 10 real problem events**. For safety, missing problems is the
  worst mistake, and AERIS rarely does that.
- **Weakness:** it raises some **false alarms**, so engineers should double-check alarms. AERIS is an
  assistant that points engineers to the right places, not a replacement for them.
- **Why per-reading scores look lower:** they're strict about the exact start and end of each problem.
- **Context:** detecting anomalies on unseen spacecraft sensors is a genuinely hard research problem.
  We report honest scores rather than inflated ones.

---

## 15. The AI Project Cycle and Model Lifecycle

### AI Project Cycle (how this project was built)
1. **Problem Scoping** – Engineers can't read millions of telemetry readings fast enough, so problems get
   missed. Who is affected? Space organisations. Where? Mission control. Why does it matter? Missions
   cost huge amounts of money and years of work.
2. **Data Acquisition** – NASA SMAP & MSL dataset from Kaggle.
3. **Data Exploration** – Looked at file shapes, plotted signals, found that column 0 is the telemetry,
   about 12.5% of test readings are anomalies, and NASA signals are mostly step-like.
4. **Modelling** – Linear Regression forecaster, Random Forest classifier, neural-network mini brains.
5. **Evaluation** – Train-test split, cross validation, precision, recall, F1, MSE, RMSE.
6. **Deployment** – A Flask website with login, history, charts, Gemini reports and a chatbot.

### Model Lifecycle (from the Class XII syllabus)
1. **Build** – train the model (`python aeris.py`).
2. **Test / Validate** – check it on hidden sensors.
3. **Retrain** – after testing, the general model was retrained on **all** NASA sensors so the deployed
   version learns from every example.
4. **Deploy** – save to `models/aeris.joblib` and load it in the website.

**Saving a model:** a trained model lives in computer memory. We save it to a file using **joblib**, so the
website can load it instantly instead of retraining every time. Running `python aeris.py` again does
nothing if a model already exists; `python aeris.py --retrain` forces a fresh training.

---

## 16. Generative AI: how Gemini writes the report

### Two kinds of AI in this project
- **Predictive / discriminative AI (AERIS):** looks at data and makes a decision (normal or anomaly).
- **Generative AI (Gemini):** creates new content, like text, images or code.

### What is a Large Language Model (LLM)?
An LLM is a huge neural network trained on enormous amounts of text. It learned to **predict the next word**
so well that it can write essays, explain topics and answer questions.

**Example:** given "The spacecraft is in…", it predicts likely next words, such as "danger" or "good health",
based on context.

**Gemini** is Google's LLM. ChatGPT and Claude are other LLMs.

### What is an API?
An **API (Application Programming Interface)** is a way for one program to talk to another over the internet.

**Example:** a restaurant waiter. You (our website) give your order (a question) to the waiter (the API), who
takes it to the kitchen (Google's servers) and brings back the food (Gemini's answer). The **API key** in the
`.env` file is like a membership card that proves we're allowed to order.

### What is a prompt?
A **prompt** is the instruction we give an LLM. Our report prompt basically says:

> "You are a report writer. Here are the analysis results (status, health score, events, metrics). Write a
> report a child could understand, with these sections: Verdict, What we checked, What AERIS found, … Do not
> invent numbers."

Writing good prompts is called **prompt engineering**.

### Important: Gemini does NOT detect anomalies
AERIS does all the detection. Gemini only **translates AERIS's results into simple language**. This is a
deliberate design choice: the decision comes from a model we trained and tested, and the explanation comes
from an LLM.

### The chatbot
The floating purple button opens a chat with Gemini. Before each question, the website quietly gives Gemini
**context**: a description of the platform, your recent analyses and the analysis you're viewing. That's why
it can answer "Is this spacecraft in danger?" correctly.

### Reliability features
- The results page opens immediately; the Gemini report loads in the background.
- If Gemini is slow or busy, it gives up after a time limit, tries a backup model, and if needed shows a
  built-in report with a "Try Gemini again" button. **The website never gets stuck.**

---

## 17. The website: what happens when you press Analyze

```
You upload a file
      │
      ▼
1. File converted to CSV (saved in uploads/)
      │
      ▼
2. Sensor columns found (time columns, counters and on/off flags skipped)
      │
      ▼
3. Is it a NASA file? (numbers compared with NASA's files)
      │  yes → expert mode (mini brain)     no → general mode
      ▼
4. AERIS: forecast → 27 clues → Random Forest → risk → alarms → status & health
      │
      ▼
5. Matplotlib draws the charts
      │
      ▼
6. Everything saved in the Neon database (your History)
      │
      ▼
7. Results page opens → Gemini writes the report in the background
```

### The pieces
- **Frontend** (what you see): HTML (structure), CSS (design: black and purple), JavaScript (interactions
  such as drag-and-drop, the chatbot and loading the report).
- **Backend** (the engine): **Flask**, a Python web framework. It handles login, uploads, analysis and the
  database.
- **Database:** **Neon**, a cloud PostgreSQL database. It stores each analysis (status, charts, report,
  summary) so History works. (If Neon can't be reached, the app falls back to a local file so it still works.)
- **Login:** accounts are pre-created in the `.env` file. There is no public sign-up, because the platform is
  meant for space organisations.
- **`.env` file:** keeps secrets (database link, Gemini key, passwords) out of the code.

---

## 18. Storytelling with data: reading the charts

Numbers alone are hard to understand; pictures tell the story. This is **data storytelling** (Class XII Unit 3).
For every sensor, the results page shows a 3-panel chart:

1. **Sensor readings** – the purple line is the data. **Red zones** are where AERIS found anomalies. For
   NASA files, **yellow bars** on top show where NASA's experts marked problems, so you can see at a glance
   whether AERIS agrees with the experts.
2. **AERIS risk score** – a line from 0 to 1. The **dashed red line** is the alarm threshold (0.4). Above it
   means an alarm.
3. **Forecast surprise** – how far each reading was from the forecaster's guess. Spikes here are the "surprises".

Also shown:
- **Health bars** for every sensor (if the file has several sensors).
- A **confusion matrix** chart for NASA files (AERIS's answers vs the experts' answers).

**Why a line chart?** Telemetry is numbers changing over time, and line charts are the best way to show
change over time (the syllabus recommends line and bar charts for numeric data).

**Charts made during training** (in `models/reports/`): model comparison, confusion matrix and feature
importance (which clues the Random Forest relied on most).

---

## 19. AI ethics in this project

- **Honesty / no misleading results:** we report real test scores on unseen data, not inflated ones. The
  page says exactly how each score was checked.
- **Transparency / explainability:** the charts show *where* and *why* AERIS raised alarms; the health score
  formula is simple and public.
- **Human in the loop:** AERIS assists engineers; final decisions stay with humans, because AI can make
  mistakes (false alarms and misses).
- **Privacy and security:** login-only access; secrets are kept in `.env` and not in the code.
- **Accessibility and inclusion:** text size controls, high contrast, easy-reading spacing, reduced motion,
  read-aloud reports, keyboard navigation and screen-reader labels.
- **Responsible use of Generative AI:** Gemini is told never to invent numbers and only to explain AERIS's results.

---

## 20. Tools and technologies

| Tool | What it is | What it does here |
|---|---|---|
| Python | Programming language | Everything in the backend |
| NumPy | Library for fast number arrays | Reading `.npy` files, maths |
| Pandas | Library for tables (DataFrames) | Reading/writing CSV and Excel, rolling statistics |
| Scikit-learn | Machine learning library | Linear Regression, Random Forest, Decision Tree, Logistic Regression, MLP, metrics, cross validation |
| Matplotlib | Charting library | All charts |
| Joblib | Saving Python objects | Saving and loading the trained model |
| Flask | Web framework | The website backend |
| HTML / CSS / JS | Web languages | The website frontend |
| Neon (PostgreSQL) | Cloud database | Saving history |
| psycopg2 | Database connector | Talking to Neon from Python |
| Google Gemini (google-genai) | Large Language Model | Reports and chatbot |
| python-dotenv | Settings loader | Reading the `.env` file |
| openpyxl / xlrd | Excel readers | Reading `.xlsx` / `.xls` uploads |

---

## 21. Glossary

- **Accuracy** – share of all predictions that were correct.
- **Activation function** – a simple rule inside a neuron that decides its output.
- **Algorithm** – a step-by-step method for solving a problem.
- **Anomaly** – something unusual compared with normal behaviour.
- **API** – a way for programs to talk to each other.
- **Backpropagation** – how a neural network adjusts its weights to reduce error.
- **Baseline** – the "normal" reference that new data is compared against.
- **Class imbalance** – when one category (normal) is much more common than another (anomaly).
- **Classification** – predicting a category.
- **Confusion matrix** – the table of TP, FP, FN and TN.
- **Cross validation** – testing a model several times on different parts of the data.
- **CSV** – a simple text table format.
- **Data augmentation** – creating extra training examples.
- **Dataset** – a collection of data used to train and test.
- **Decision tree** – a flowchart of yes/no questions.
- **Deep learning** – machine learning with neural networks.
- **Deployment** – putting a trained model into real use.
- **Event** – a continuous period of anomalous readings.
- **F1 score** – the balance of precision and recall.
- **Feature** – a clue/input given to a model.
- **Feature engineering** – creating good features.
- **Forecast** – a prediction of a future value.
- **Generative AI** – AI that creates new content.
- **Hyperparameter** – a setting chosen before training.
- **Inference** – using a trained model to make predictions.
- **Label** – the correct answer for an example.
- **LLM** – Large Language Model, like Gemini.
- **MAE / MSE / RMSE / MAPE** – ways to measure prediction errors.
- **Model** – what a computer builds after learning from data.
- **Neural network** – layers of connected artificial neurons.
- **Normalisation / Scaling** – putting values on a common range (−1 to 1).
- **Overfitting** – memorising training data instead of learning.
- **Precision** – share of alarms that were right.
- **Prompt** – the instruction given to an LLM.
- **Random Forest** – many decision trees voting together.
- **Recall** – share of real problems that were found.
- **Regression** – predicting a number.
- **Supervised learning** – learning with answers.
- **Telemetry** – sensor data sent from far away.
- **Test set** – hidden data used for the final check.
- **Threshold** – the cut-off value that turns a probability into a yes/no decision.
- **Training** – the learning phase.
- **Underfitting** – a model too simple to learn.
- **Unsupervised learning** – learning without answers.
- **Z-score** – how many standard deviations a value is from the average.

---

## 22. Viva questions and answers

**About the project**

**Q1. What is your project?**
A website that uses an AI model, AERIS, to analyse spacecraft telemetry and tell whether the spacecraft is
healthy, needs attention, or is in danger.

**Q2. What problem does it solve?**
Spacecraft send too much data for humans to check quickly. AERIS checks every reading in seconds and points
engineers to problems.

**Q3. What is telemetry?**
Sensor data sent from a spacecraft back to Earth.

**Q4. Where did you get your data?**
The NASA SMAP & MSL anomaly detection dataset on Kaggle: 82 sensors and 105 anomalies labelled by NASA engineers.

**Q5. Why do you convert files to CSV?**
CSV is a simple, universal table format. Converting everything to CSV means the analysis only needs to handle
one format, and Pandas reads it easily.

**About the AI**

**Q6. Which algorithms did you use?**
Linear Regression (forecasting), Random Forest (anomaly classification), and small neural networks (one mini
brain per NASA sensor). Logistic Regression and a Decision Tree were tested and compared.

**Q7. Why Random Forest?**
It had the best F1 score (0.482) in 5-fold cross validation, compared with Logistic Regression (0.389) and
a Decision Tree (0.374).

**Q8. What is a Random Forest?**
Many decision trees, each trained on random parts of the data, that vote. The majority vote is the answer.

**Q9. Is your project supervised or unsupervised?**
Both. The Random Forest is supervised (it learns from NASA's labels). The mini brains learn only normal
behaviour without labels, which is unsupervised in spirit.

**Q10. What is the difference between regression and classification?**
Regression predicts a number (the next reading); classification predicts a category (normal or anomaly).

**Q11. What are features? Give examples.**
Clues given to the model. Examples: rolling average, standard deviation, z-score, rarity of the value, and the
forecast error.

**Q12. What is a neural network?**
A model made of layers of connected neurons. Each neuron multiplies inputs by weights, adds them up and applies
an activation function. It learns by adjusting the weights to reduce errors.

**Q13. What is a hyperparameter? Give examples.**
A setting chosen before training. Examples: 120 trees, 30 past readings, alarm threshold 0.4, 5 folds.

**About evaluation**

**Q14. What is a train-test split? How did you do it?**
Keeping some data hidden for the final test. I split by sensor: 64 sensors for training and 17 hidden, so
the test uses sensors the model never saw.

**Q15. What is cross validation?**
Splitting data into K parts, training on K−1 and testing on the remaining one, K times, then averaging. I used K = 5.

**Q16. What is overfitting? How did you prevent it?**
Memorising training data instead of learning patterns. I prevented it with a by-sensor train-test split,
cross validation, limited tree depth, data augmentation and early stopping.

**Q17. Define precision and recall.**
Precision = TP ÷ (TP + FP), how many alarms were right.
Recall = TP ÷ (TP + FN), how many real problems were found.

**Q18. Why not just use accuracy?**
Anomalies are rare. A model that always says "normal" gets high accuracy but finds nothing. Precision, recall
and F1 show the real performance.

**Q19. What is the F1 score?**
2 × P × R ÷ (P + R). It balances precision and recall; 1.0 is perfect.

**Q20. What do MSE and RMSE tell you?**
How far the forecaster's guesses are from the real readings. RMSE is the square root of MSE and is in the
same units as the data. Smaller is better.

**Q21. What are your model's results?**
On NASA sensors (expert mode) it catches 97 of 105 real anomaly events (92%), with F1 0.54, precision 0.52
and recall 0.57.

**Q22. Is your model perfect?**
No. It catches about 9 out of 10 problem events but also raises some false alarms, so engineers should verify
alarms. It's a helper, not a replacement.

**Q23. What is a confusion matrix?**
A 2×2 table of true positives, false positives, false negatives and true negatives.

**About Gemini and the website**

**Q24. What does Gemini do?**
It turns AERIS's results into an easy report and summary, and powers the chatbot. It does not detect anomalies.

**Q25. What is an LLM?**
A Large Language Model: a huge neural network trained on text to predict the next word, so it can write and
answer questions.

**Q26. What happens if Gemini fails?**
The results and metrics still show. A built-in report is displayed with a "Try Gemini again" button.

**Q27. What is the model lifecycle?**
Build → Test → Retrain → Deploy. AERIS was trained, tested on hidden sensors, retrained on all data and
deployed in the website.

**Q28. What are the ethical considerations?**
Honest metrics, transparency through charts, a human in the loop, privacy (login only, secrets in `.env`)
and accessibility features.

**Q29. Future scope?**
Real-time live monitoring, more spacecraft and sensors, deep learning models such as LSTM, and alerts by
email or SMS.

**Q30. Which charts did you use and why?**
Line charts for readings, risk score and surprise, because telemetry changes over time. Bar charts for health
scores and model comparison, and a confusion matrix for checking against experts.

---

## 23. Syllabus map

| Syllabus topic (CBSE AI 843) | Where it appears in this project | Chapter |
|---|---|---|
| Understanding the problem / problem scoping | Telemetry overload problem | 1, 15 |
| Data requirements and collection | NASA SMAP & MSL dataset, CSV conversion | 4 |
| Modelling approach | Three-stage AERIS design | 6, 8, 10 |
| Commonly used algorithms | Linear Regression, Logistic Regression, Decision Tree, Random Forest, Neural Network | 6, 8, 9, 10 |
| Train-test split | 64 training / 17 hidden sensors | 11 |
| Cross validation | 5-fold Group cross validation, cross-fitting | 11 |
| MSE, RMSE, MAPE | Forecaster metrics on every analysis | 13 |
| Precision, Recall, F1, Confusion matrix | Classification metrics on every analysis | 13 |
| Hyperparameters | `CONFIG` in `aeris.py` | 11 |
| Model lifecycle: build, deploy, retrain | Train → test → retrain → website | 15 |
| Storytelling through data | Matplotlib charts + Gemini report | 16, 18 |
| Ethics of AI | Honest metrics, human in the loop, accessibility | 19 |

---

*You built this project with the help of AI tools. That is exactly how modern engineers work. This guide is so
you can also understand and explain every idea inside it. Good luck with your viva! 🚀*

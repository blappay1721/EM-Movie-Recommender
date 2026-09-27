# Movie Recommendations with EM

A movie recommender built on a latent-class (naive Bayes) model and trained with Expectation-Maximization, written from scratch in NumPy. It also includes an interactive survey: rate the movies you've seen and it recommends the ones you haven't.

The model assumes every student is one of **k = 4 hidden movie-goer types**, and each type has its own probability of liking each movie. EM learns the types from a 288 × 60 ratings matrix where about 40% of the entries are missing. It then fills in the gaps for any new person by working out which types they most resemble.

## How it works, in plain English

**The idea.** People's movie tastes fall into a few broad groups. Someone who loved *The Avengers* probably liked *Iron Man 3* too. Nobody labeled these groups in the survey, though. Each student only said yes, no, or "haven't seen it" for 60 movies. The model's job is to discover the groups from the answers alone.

**What the model learns.** It learns two things:

1. **How common each type is.** For example, "type 4 covers 42% of students."
2. **Each type's taste profile.** For every movie, it learns how likely someone of that type is to recommend it. For example, "a type 2 person recommends *Avengers: Infinity War* about 93% of the time."

**The chicken-and-egg problem.** If you knew which type each student was, working out each type's taste would be easy: just average what the members of that group said. And if you knew each type's taste, you could easily guess which type a student is by seeing whose taste their answers match best. You know neither at the start. Expectation-Maximization (EM) gets around this by guessing and then refining, over and over:

- **Start with a rough guess** of the taste profiles. Here that's the random starting values provided with the assignment.
- **E-step (Expectation), "who's who?"** For each student, compare their answers against every type's taste profile and assign them a *soft* membership, such as "80% type 4, 20% type 1." Movies the student hasn't seen are simply left out of the comparison.
- **M-step (Maximization), "what does each group like?"** Recompute each type's taste profile from the students assigned to it, weighting each student by their membership. Students who are 80% type 4 count 80% toward type 4's profile.
- **Repeat.** Better memberships lead to better profiles, which lead to better memberships. Each round is guaranteed to fit the data at least as well as the last one. After a few dozen rounds the numbers stop changing, and the notebook runs 256 to be safe.

**Turning it into recommendations.** Once the types are learned, recommending for a new person is just the E-step followed by a weighted average:

1. You rate the movies you've seen.
2. The model works out your type mix from those ratings, for example "74% type 2, 26% type 3."
3. For each movie you *haven't* seen, it blends the types' opinions using your mix. With the mix above, type 2's opinion gets 74% of the weight and type 3's gets 26%. The result is the chance you'd recommend that movie.
4. Your unseen movies are sorted by that chance, and the top ones become your recommendations.

This is why the results are personal, unlike a plain "most popular" list. Two people who both haven't seen *Ex Machina* can get very different predictions for it depending on which group their other ratings place them in. The more movies you rate, the more confident the model is about your type, and the sharper your recommendations get.

## Try it

Run all cells, then scroll to **Try it yourself**. Each of the 60 survey movies gets a **Yes / No / ?** toggle:

- **Yes**: you'd recommend it
- **No**: you wouldn't
- **?**: you haven't seen it

Press **Recommend** to see your inferred type mix and your top 10 unseen movies ranked by expected rating. The form needs a live Jupyter kernel because GitHub's notebook viewer can't run widgets.

## Notebook

[`em_movie_recommender.ipynb`](em_movie_recommender.ipynb) covers:

- a mean-popularity baseline, where everyone gets the same list
- the model, plus the E-step and M-step derivations
- vectorized EM in log space, where each iteration is a few matrix products with log-sum-exp for stability
- a log-likelihood curve, with a check that it never decreases
- the four learned types and the movies each one loves and skips
- personal recommendations for one row of the survey
- the interactive survey form (`ipywidgets`)

```bash
pip install -r requirements.txt
jupyter notebook em_movie_recommender.ipynb
```

## Results

The normalized log-likelihood rises from −33.41 to **−18.0471** after 256 iterations, which matches the course reference values to four decimal places.

## Data

`data/` holds an anonymized survey, where each rating is `1` (recommend), `0` (don't recommend), or `?` (not seen). It also holds the provided initial values for P(Z = i) and P(R_j = 1 | Z = i).

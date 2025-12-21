# MakeMoreV1
# My Notes on Andrej Karpathy's Makemore Series

Implementation and experiments following Andrej Karpathy's excellent "Makemore" tutorial series  
(Part of his Neural Networks: Zero to Hero playlist).

These are my personal notes, insights, and explanations from building the models step-by-step.

## Part 1: Bigram Model

### In W(27, 27)

- row i: probability distribution of the next character given that the previous char was "i"
- col j: how often char "j" appeared as the second char in a bigram, across the whole dataset.

### What does "F.one_hot(xs, num_classes=27).float()" do?

- never feed raw integers that represent categories into a neural net and multiply them by weights -- you are lying to the network about distances that don't exist!!
- it takes integers in "xs" and turns them into vectors of length 27, with all zeros except a single 1 at the position corresponding to the index.

### What does "xenc @ W" do?

- for each character that each row in "xenc" represents, we are copying that character's row from the weight matrix W and we get 27 numbers (logits). these 27 numbers are the learned "strengths" for every possible next character.

### What does "logits.exp()" do?

- exponentiates all 27 numbers -> turns them into positive "fake counts".

### Why "keepdim=True" in "counts / counts.sum(1, keepdim=True)"?

- by default "keepdim" is set to "False" for some reason.
- "counts.sum(1, keepdim=True)" -> returns a tensor of size (27, 1) with each row containing the sum of that row.
- when we are dividing (27, 27) tensor with a (27, 1) tensor, torch copies the (27, 1) tensor over, so we get a (27, 27) tensor.
- if we don't write "keepdim=True" torch internally will create a (1, 27) tensor and stretch it to a (27, 27) tensor which will have the WRONG values after division takes place.

### Why "0.01 * (W ^ 2).mean()"?

- first of all this is our L2 regularization (also called weight decay). the model learns almost the same probabilities, but with much smaller weights.
- "(W ^ 2).mean()" -> this gives a single number measuring how large the weights are on average.
- * 0.01" -> is the regularization strength, its small because the main goal is to minimize the negative log-likelihood, we only want a gentle nudge toward smaller weights.

### How the Model Works

- given the previous character, here are the 27 probabilities of what comes next.
- pick one, repeat.
- it has zero idea about:
  - how long the name already is.
  - what char appeared 2, 3 or 10 steps ago.
  - whether we are at the beginning, middle, or the end of the name.
  - any spelling rules beyond "what usually follows this single letter."

- that's why our lowest negative log-likelihood will be ~2.42, because it's the ceiling (best possible) for any bigram model on this dataset.
- you cannot get lower than ~2.42 with only one character of memory, no matter how perfectly you train it.
- every single bigram in the entire dataset is treated as a completely independent training example.

### "torch.multinomial()" note

- returns a tensor so to get the value we do -> ".item()"
that's why our lowest negative log-likelihood will be ~2.42, because it's the ceiling (best possible) for any bigram model on this dataset.
you cannot get lower than ~2.42 with only one character of memory, no matter how perfectly you train it.
every single bigram in the entire dataset is treated as a completely independent training example.

"torch.multinomial()" -> returns a tensor so to get the value we do -> ".item()"

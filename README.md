# 2l-dr
tl-dr (abstractive text summarization) take 2

## How to run
We have shellscripts in ./scripts that giveyou examples of how to run the model
Obviously, you can't run the full thing if you don't have the Gigaword dataset.
But we did put some toy datasets in here.  Not that I guarantee you'll learn
anything on them... 

We did also build in an option to just run and generate summaries, but
the project rubric says to not put model checkpoints.  So this feature is moot.

## Example results
Actual headline: dpp chairman touts 'three direct links' with mainland china
Generated headline: dpp chairman says china will improve ties with taiwan <\s>

Actual headline: two palestinians killed in gaza strip clash with israeli army (updates with toll downgraded by israel, adds jericho arrest)
Generated headline: two palestinians killed in gaza strip <\s>

### Why is it called 2l-dr?
We've tried tackling the problem of text summarization before
(our intial attempt was calle tl-dr), and this is our second attempt.  

### Models
We support a GRU encoder-decoder model, and a QRNNenc + RNNdec model.
We used to run a QRNN encoder-decoder model, but we swapped the decoder out.
But the code for it lives on.

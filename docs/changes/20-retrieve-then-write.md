## What is left

Ask searches a released dataset and returns hits. It does not let the customer's from-scratch model pick the passage, and it does not let the local Qwen runner write the sentence. `cited_answer` already drops a reply that moves a line onto another file. Ask does not call it.

## Change

- Ask retrieves with the promoted from-scratch model when one is served, and with keyword search when none is.
- The local runner writes from those hits only, through `cited_answer`.
- A line number printed on a file it does not belong to fails. The stored answer is the hit list, not the miscited sentence.
- The runner is the on-prem adapter. The question is not sent to a public host.

## Proof

- A hit for file A line 1, and a completion that puts that line on file B, is stored as the hit list and marked failed.
- A completion that quotes file A line 1 is stored as the written answer.
- Ask with no promoted from-scratch model still returns keyword hits.

## Out of scope

Do not mix Qwen weights into the from-scratch model. Do not run Ask over the 12,721-record corpus.

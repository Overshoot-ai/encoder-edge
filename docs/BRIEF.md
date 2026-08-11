Cross-device Encoder - Decoder Disagg
Ali Tayeb

Motivation:
The goal is to separate the vision encoder from the LLM decoder and have the former run on the client and the latter in the GPU host
The goals behind this are multi-fold:
Privacy: this is the number one reason. Clients who don’t want to send their actual picture can just run the encoder on the client and send us the embeddings. This will allow them to still run the big model but save on embeddings.
Potential reduction in bandwidth in the future: In the present, embeddings that come out of the vision encoder are actually quite big (roughly 300kb - 1.4Mb depending on the number of tokens), meanwhile the frame itself is only around 50 - 70kb. However, techniques like token merging and token pruning can reduce the number of vision tokens required by 80% while keeping the same performance, and that’s where the savings in bandwidth will come from

Step 1: (no code)
Look up the literature and who’s doing it today
ChatGPT and other AI models are usually not good and finding the latest stuff so use alphaxiv for latest research and just regular google search and X and see who did it recently
Answer the following questions:
What have people tried in the past?
What’s the latest attempt at this?
Is there an inference platform that offers it?
Are there open-source implementations of this?
What are the pros and cons of it?
Put together a table with each VLM, and the encoder size and the FLOPs required for a single frame
Note that some models like a version of Gemma have really tiny encoders, like literally just an embedding layers (it’s called unified vision encoder in the literature)

Step 2:
The table above will give us a good understanding on the compute and memory requirements for this for different models.
Can it run on the browser? Or a Jetson? Or a mac … etc
We pick one model and implement it for it. We will pick a single use-case. It could be anything from a conversational AI or event detection. We can decide this together.
The implementation should be lean and we should be able to read the code easily

Step 3:
Once we implement it, we want to evaluate it to make sure it works correctly
And then benchmark it to understand how much slower it is than sending the image / video
Then we can write a technical blog about it and potentially open-source it

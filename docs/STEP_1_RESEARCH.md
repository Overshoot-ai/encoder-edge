# Step 1 Research: Cross-Device Vision-Language Model Encoder/Decoder Disaggregation

Research checked: 2026-07-29

## Scope and assumed reader

This report assumes the reader understands that a neural network maps input numbers to output numbers, but does not assume prior knowledge of a vision-language model (VLM), model serving, compression, or network security. A VLM accepts an image plus text and generates text.

The project being evaluated is specifically this:

- One client owns an image or video frame and a text question.
- The client should avoid sending the original image to the service.
- The client runs the image-processing part of one specific VLM.
- A graphics processing unit (GPU) server runs the language-generation part of that same VLM.
- The client and server exchange a model-specific tensor, meaning a multidimensional array of numbers, not a generic image embedding that works with every model.
- The first privacy objective is **raw-pixel minimization**: the server should not receive the original image file. An ordinary plaintext split VLM does not hide the task-relevant meaning represented in those numbers, because the server directly reads them to answer.
- The image-size baseline of 50-70 KB comes from `docs/BRIEF.md`. It is a project estimate, not a universal JPEG/WebP size. Actual compressed size depends on resolution, format, quality setting, and image content.

Search scope: alphaXiv/arXiv papers, paper references, official model documentation, GitHub repositories, vLLM, llm-d, TensorRT-LLM, NVIDIA Dynamo, and hosted inference providers including Modal, Baseten, Fireworks, Together AI, and Hugging Face Inference Providers were checked through 2026-07-29. "Latest," "smallest," and "no implementation found" mean within this search, not a proof that no private, commercial, unindexed, or newly published system exists.

## Prerequisites: how a VLM processes one image

### The normal unsplit pipeline

A **vision-language model (VLM)** accepts visual input plus text and generates text. A common VLM has three separately identifiable parts:

1. **Image processor:** Ordinary code resizes the image, converts red-green-blue (RGB) pixel bytes to floating-point numbers, normalizes the color channels, and arranges the result into the tensor shape expected by the model. This code has no learned reasoning ability, but its exact resize and normalization settings must match training.
2. **Vision encoder:** A learned neural network converts image patches into a sequence of vectors. In many models this is a Vision Transformer, abbreviated **ViT**. Each output vector summarizes some visual information.
3. **Projector or connector:** A learned layer changes each vision-encoder vector to the width and distribution expected by the language model. An encoder may output vectors of width 1,024 while the language model expects width 4,096.
4. **Language model (LLM):** The LLM consumes the projected visual vectors together with text-token vectors. During **prefill**, it reads the complete question and visual sequence once and builds an attention memory called a **key-value (KV) cache**. During **decode**, it generates answer tokens one at a time while reusing that cache.

```text
image file
  -> resize and normalize
  -> vision encoder
  -> projector
  -> insert visual vectors at the image position in the text prompt
  -> LLM prefill
  -> LLM autoregressive decode
  -> answer text
```

### What the project changes

The split moves the first stages to the client:

```text
CLIENT                                                        SERVER

image file                                                    no image file
  -> exact resize/normalize                                     receives prompt + tensor
  -> exact vision encoder                                       -> validate metadata and values
  -> optional projector                                         -> place visual vectors in prompt
  -> optional pruning/quantization/compression                   -> LLM prefill creates KV cache
  -> binary serialization                                       -> LLM decode generates answer
  -> authenticated network request --------------------------->  -> return answer text
```

The text question may be sent before, after, or with the visual tensor. A query-independent encoder can encode before the question is known and its output can be reused. A query-conditioned compressor such as QueCC needs the question, so its output cannot automatically be reused for a different question.

### The split contract

The client cannot combine any convenient image encoder with any convenient LLM. Both sides must implement one compatible trained model. Every request must identify or imply all of the following:

| Contract field | Why it must match |
|---|---|
| Model and weights revision | Different learned weights produce incompatible numerical representations. |
| Image-processor revision | A different resize, crop, RGB order, or normalization changes every encoder value. |
| Split point | The server must know which exact model layer produced the tensor and which layer runs next. |
| Projector location and revision | The server must know whether vectors are still in encoder width or already in LLM width. |
| Tensor shape and axis order | For example, `[1, 576, 4096]` means batch size 1, 576 visual tokens, and 4,096 values per token. |
| Numeric format | FP32 and FP16 are 32-bit and 16-bit floating point; BF16 is a different 16-bit floating-point layout; INT8 is an 8-bit integer. Packed 4-bit values use half a byte before metadata. Each requires different decoding. |
| Token and position order | Reordering visual positions changes spatial meaning. Dynamic-resolution models may need image-grid metadata. |
| Compression/codebook revision | Quantized or compressed values are meaningless without the matching scales, codebook, or decoder. |

### Names for the values moving through the model

An image first becomes **patches**, small pixel blocks such as 14x14 RGB squares. A **visual token** is one vector representing visual information; 576 tokens of width 4,096 form a matrix containing `576 * 4096` values. A **feature** or **activation tensor** is the general name for a numeric intermediate produced at a named layer. The **encoder output** is specifically the final feature tensor from the vision encoder.

After the projector maps an encoder vector to the language model's width, this report calls it a **projected embedding**. A **latent** is different: it is a compressed learned representation that normally needs a matching decoder before the VLM can consume it. A **codebook index** is smaller still, because it is only an integer selecting a learned vector from a shared table. A **soft token** is a continuous vector inserted where the language model normally receives a text-token embedding; it is not a discrete vocabulary ID.

Model names are introduced where they become relevant. CLIP and SigLIP are families of pretrained image encoders. LLaVA is a VLM family that connects one of these encoders to a language model through a projector. Visual question answering (VQA) means answering a text question about an image, while optical character recognition (OCR) means reading text visible inside an image.

Numeric formats are also defined at first use. FP16 means a 16-bit floating-point number and ideally costs two bytes per value. INT8 means an 8-bit integer and costs one byte before its quantization scale metadata. A packed 4-bit value costs half a byte before metadata and alignment. A floating-point operation (FLOP) is one arithmetic operation; this report counts a multiply and add as two operations. A GFLOP is one billion operations and a TFLOP is one trillion. These measure work for one inference, not how many seconds a particular device takes.

For serving results, **end-to-end (E2E) latency** means wall-clock time from request start until the complete answer arrives. **Time to first token (TTFT)** ends when the first answer token arrives. **Throughput** is completed requests or generated tokens per second under sustained load, so throughput can improve even when one isolated request does not. A P99 latency is the value below which 99% of requests finish and therefore describes the slow tail. A service-level objective (SLO) is a target such as "99% of requests start responding within two seconds." Network round-trip time (RTT) is the time for a message and reply. Remote direct memory access (RDMA) is a fast datacenter memory-transfer mechanism, not normal Internet transport.

## One concrete tensor example

LLaVA-1.5 makes the size problem easy to see:

1. The image processor resizes one image to 336x336.
2. CLIP ViT-L/14 divides it into 14x14 patches: `336 / 14 = 24` patches along each axis, or `24 * 24 = 576` image patches.
3. The vision encoder emits roughly 576 vectors of width 1,024.
4. The projector maps each vector from width 1,024 to the Vicuna LLM width of 4,096.
5. Sending the post-projector tensor as FP16 costs `576 * 4096 * 2 = 4,718,592` bytes, or 4.50 MiB, before headers or encryption. One mebibyte (MiB) is 1,048,576 bytes; one kibibyte (KiB), used below for smaller values, is 1,024 bytes.
6. Sending the pre-projector tensor as FP16 costs `576 * 1024 * 2 = 1,179,648` bytes, or 1.125 MiB. This is smaller, but the server must then run the exact projector.

Both tensors are much larger than the brief's 50-70 KB compressed-image estimate. Therefore "send embeddings instead of pixels" saves image handling and may improve raw-pixel privacy, but it does not save bandwidth unless the tensor is also pruned, quantized, or encoded as compact indices.

## The simplest way to think about the proposal

- **Privacy:** The server no longer receives the original image file, which is useful data minimization. It still receives vectors describing the objects, text, people, and scene well enough to answer, so those vectors remain sensitive data.
- **Bandwidth:** Moving the encoder to the client does not by itself reduce bytes. Ordinary floating-point encoder outputs are often larger than JPEG or WebP. Bandwidth improves only after choosing a naturally compact model or adding strong token reduction and numeric compression.
- **Compute:** Vision-encoder work does not disappear; it moves from the server to the client. This can improve total server throughput when many clients encode in parallel, but adds client latency, memory use, battery use, heat, and model download size.
- **Implementation:** vLLM and TensorRT-LLM can accept precomputed visual embeddings, and several stacks can separate encoder and language-model workers inside a datacenter. No hosted provider checked offers client-generated VLM embeddings as a turnkey API. Modal and Baseten can host a custom vLLM or Python/container implementation, but the project must still supply the embedding contract and validation gateway. These embedding APIs are model-specific and not hardened public-Internet contracts: vLLM explicitly says to enable its endpoint only for trusted users because malformed shapes may crash the engine.
- **Best first experiment:** SmolVLM-256M has a small `64 x 576` split tensor and a published embedded-device test. LLaVA-1.5 is easier to understand and instrument but sends multi-megabyte tensors unless compressed. Both should be compared against sending the exact same test images as WebP/JPEG.

## Executive answer

The proposed architecture already exists in research and is beginning to appear in open-source serving stacks:

1. **Direct systems already exist.** Distributed VLMs (2025) runs the complete vision encoder and projector on edge devices and reports up to 33.54% higher throughput by overlapping edge encoding with server decoding. TOFC (2025) runs the complete encoder and learned feature compressor on the device, then sends an entropy-coded pre-projector representation to the server.
2. **The latest qualifying deployment proposal found is Co-VStream (June 2026).** It runs VideoLLaMA3-7B's complete 0.4B vision encoder at the edge, condenses features, and sends them with edge-generated captions to a cloud VideoLLaMA3-7B reasoner. It is less pure than a simple encoder/LLM split because the caption side channel and cloud entity graph materially contribute to accuracy. Progressive Semantic Communication (April 2026) remains the latest cleaner physical testbed found: an NXP i.MX95 runs the complete SmolVLM encoder and an RTX 2080 Super runs its LLM.
3. **TOFC is the strongest direct feature-compression result found.** Its current paper reports up to 52% less transmitted data and 63% lower system latency than the learned image codec ELIC at matched task performance. Its full encoder runs on the device; the server reconstructs features, not pixels, before running the projector and LLM.
4. **Do not rely on AlignedVQ's headline network numbers.** The 2024 version reported a 0.845 KiB payload, 1,365x compression, and 2-15x speedup, but the authors stated in the April 2026 revision that residual links were omitted and that all compressed-size and transmission-latency results were affected. No corrected network figures were supplied. AlignedVQ is also an intra-encoder split, not a complete client-side encoder.
5. **Serving support exists, but not as a turnkey hosted product.** vLLM's OpenAI-compatible Chat Completions API accepts base64 precomputed multimodal embeddings when started with `--enable-mm-embeds`; TensorRT-LLM supports externally computed embedding tensors as well as a separate standalone-encoder handle path. Modal can deploy a customized vLLM service, and Baseten can deploy vLLM in a custom container or accept arbitrary JSON through Truss. Their standard model APIs, and the checked Fireworks, Together AI, and Hugging Face provider APIs, do not expose client-generated VLM visual tensors. llm-d documents trusted-cluster encoder disaggregation, not external tensor ingress.
6. **Embeddings are not equivalent to privacy.** Encoder outputs can leak captions and labels, and encoder-free visual tokens can leak readable text and image structure. The server still receives the semantic content needed to answer the question. This architecture prevents routine receipt of raw pixels, but does not provide cryptographic or formal privacy by itself.
7. **The network link must provide integrity as well as confidentiality.** A July 2026 paper changed only 10% of in-transit vision tokens and reduced Qwen2.5-VL-72B accuracy on MMBench from 88.39% to 0.08%.
8. **Raw FP16 embeddings usually lose to JPEG/WebP on bandwidth.** For a 4,096-wide decoder embedding, a 50-70 KB image is beaten only at about 6-8 FP16 tokens, 12-17 INT8 tokens, or 25-35 4-bit tokens. Practical bandwidth savings require aggressive token reduction plus quantization, a learned codec, or discrete codebook indices.

## What counts as the same idea?

This report calls a system an **exact match** only when all three conditions hold:

1. The client runs the target VLM's complete vision encoder.
2. The server receives a representation that the remaining target VLM can consume without reconstructing an image and rerunning vision encoding.
3. The target VLM's LLM runs on the server.

Everything else is related work rather than the same deployment.

| Design | Simple example | What crosses the network? | Exact match? |
|---|---|---|---|
| Whole-encoder split | Client runs CLIP; server runs LLaVA's projector and Vicuna | CLIP encoder output | Yes |
| Whole-encoder-plus-projector split | Client runs CLIP and projector; server runs Vicuna | LLM-width projected embeddings | Yes |
| Intra-encoder split | Client runs the first CLIP block; server runs the rest | An intermediate layer activation or compressed indices | No, but a close partitioning alternative |
| Datacenter encode disaggregation | One server GPU encodes; another server GPU generates | Internal cache reference or RDMA tensor | Same neural-network boundary, but it does not keep pixels on a user device |
| Model routing | Phone answers easy requests with a small VLM; cloud handles hard requests with a large VLM | Usually the original image for cloud requests | No |
| Selected-region retransmission | Client sends a small image, then a requested region-of-interest crop | Pixels | No |
| Discrete image-token transport | Client sends image-codec IDs; server reconstructs pixels | Codec indices | No, because the target VLM re-encodes the reconstructed image |
| Speculative co-inference | Client small VLM proposes answer tokens; server large VLM verifies | Draft text tokens and probability values | No |

## Evidence at a glance

The following table states what each central result actually demonstrates. A dash means the paper or public artifact did not provide enough information for this report to state the value reliably.

| Work | Client-side model stage | Transmitted object | Reported payload | Baseline and metric | Training needed for split/compression? | Public runnable exact split? |
|---|---|---|---:|---|---|---|
| Distributed VLMs | Complete vision encoder and projector | Projected visual tensor | - | Up to 33.54% more completed requests per unit time than cloud-only serving; benefit comes from overlapping different requests | No model compression training | No code found |
| Progressive Semantic Communication | Complete SmolVLM encoder plus Meta AutoEncoder compressor | Ordered compressed latent chunks | Full latent serialized in JSON is reported as about 192.3 KiB; raw JSON image is 1,024.3 KiB | 6.94 s E2E at full latent versus 9.32 s full-cloud under 1 Mbps; 120 generated tokens | Yes, train the Meta AutoEncoder only | No; repository contains only a license |
| TOFC | Complete SigLIP or CLIP encoder, feature merger, and entropy encoder | Entropy-coded merged pre-projector features | Rate varies by quality target | Up to 52% less data and 63% lower latency than ELIC at matched task performance | Yes; feature codec and model adaptation | Partial method code; cross-device transport absent |
| Co-VStream | Complete VideoLLaMA3-7B vision encoder plus temporal feature condensation; separate captioner | Condensed visual features and captions | 316.42 MB over the LVBench stream evaluation | 87.59% less communication and 38.93% versus 39.23% cloud-only LVBench accuracy | No; training-free pipeline | No public code found |
| AlignedVQ | First vision-transformer block plus vector quantizer, not the complete encoder | Codebook IDs plus residual-path data | No valid corrected total reported | Original 0.845 KiB and 2-15x claims are affected by omitted residual links, according to the April 2026 revision | Yes, one epoch; codebook, surrounding adapter layers, and small LLM adaptation weights | No public code found |
| ModServe | Dedicated datacenter image-encoder GPUs | Internal image-token tensors over remote direct memory access or ordinary Transmission Control Protocol (TCP) networking | Model dependent | 3.3-5.5x serving throughput and 25-41.3% cost reduction versus monolithic serving on 128 A100s | No accuracy-changing compression | No public source found |
| VisionZip | Complete encoder plus token selector/merger | Reduced floating-point visual-token sequence | 64 x model width at its smallest main LLaVA-1.5 setting | 94.0% normalized average benchmark score at 64 versus 576 tokens, training-free | Optional projector tuning | Public compression code, not a complete cross-device service |
| Sema | Separate Cosmos image codec, not target VLM encoder | Discrete codec IDs plus optional text extracted from the web page or OCR | 6,548 average bytes in its hybrid test | 30 generated browser screenshots at 5 Mbps/50 ms RTT; hybrid succeeded, token-only failed | No target-VLM modification | Yes, but it is a reconstruct-then-re-encode design |

## 1. What have people tried in the past?

### Direct matches

**Paper 1 (Yuyang Li, Devika Gumaste, Mehmet Kerem Turkcan, Javad Ghaderi, Gil Zussman, and Zoran Kostic):** Used almost exactly the proposed technique in **Distributed VLMs: Efficient Vision-Language Processing through Cloud-Edge Collaboration**. For request B, the edge runs the complete vision encoder and projector while the server is still generating the answer for request A. The edge sends request B's text prompt first, then sends its projected visual tensor with the same request ID. The server matches the two pieces, batches ready requests, and inserts the visual vectors into the LLM input. PAUSE/RESUME messages stop clients from adding work when the server queue is full; this is **backpressure**. The reported peak improvement of 33.54% is sustained throughput under multiple requests, not evidence that one isolated request becomes 33.54% faster. The model's mathematical output should remain unchanged because features are neither pruned nor quantized, although the paper does not provide a substantial accuracy-equivalence table. The testbed uses a Jetson Orin Nano 8 GB edge, Jetson AGX Orin 64 GB server, and gigabit switch. It evaluates LLaVA-Llama-13B, LLaVA-Qwen-4B, and Moondream2 on Visual Riddles. Models whose encoder occupies a larger fraction of total inference time gain more because more server work can overlap with edge work. Adding a second edge device stops helping once the server LLM is continuously busy. The paper does not state a reusable payload-size result, and no source implementation was found. [NSF record](https://par.nsf.gov/biblio/10639785-distributed-vlms-efficient-vision-language-processing-through-cloud-edge-collaboration) | [paper PDF](https://www.ee.columbia.edu/~jghaderi/perconai25.pdf)

**Paper 2 (Cyril Shih-Huan Hsu, Wig Yuan-Cheng Cheng, and Chrysa Papagianni):** Extended the exact split with learned, incremental compression in **Progressive Semantic Communication for Efficient Edge-Cloud Vision-Language Models**. The NXP i.MX95 runs a 4-bit SmolVLM-256M vision encoder that emits a `64 x 576` tensor. A two-layer Meta AutoEncoder, or **MetaAE**, converts that fixed tensor into four ordered chunks. The first chunk is intended to preserve coarse information; later chunks refine reconstruction. At a 25% latent-transmission level, the edge sends the first of four chunks, not 25% of arbitrary bytes or image pixels. The RTX 2080 Super server runs the matching MetaAE decoder and predicts whether reconstructed features meet a quality threshold; otherwise it requests the next chunk, adding another network round trip. Only MetaAE is trained, using 50,000 images from COCO, a common object-recognition dataset, without using their captions; SmolVLM stays frozen. The full-cloud baseline sends a 512x512x3 image encoded as base64 in JSON, reported as 1,024.3 KiB. The 100% progressive representation, serialized as JSON, is about 192.3 KiB. This is larger than the ideal 72 KiB for a `64 x 576` FP16 tensor because the paper's measured path uses a different representation and inefficient JSON serialization; MetaAE's advantage is that it can send only an ordered prefix, not that its full JSON form is smaller than an ideal binary encoder tensor. At a throttled 1 Mbps uplink with a 200 ms traffic-control latency setting and 120 generated tokens, mean E2E times are 6.94 s split, 7.98 s full-edge, and 9.32 s full-cloud. In the split path, 3.21 s is edge vision encoding, 1.62 s is network transfer, and 0.49 s is server LLM decode; MetaAE edge encoding adds about 0.01 s. The "75% reduction" at 25% chunks refers approximately to transmission delay relative to sending the full latent, not a 75% E2E improvement. **Semantic consistency** is cosine similarity between sentence embeddings of the compressed-path answer and the uncompressed SmolVLM answer. It remains above 80%, and the paper describes less than a five-percentage-point loss at 25%; this does not prove either answer is correct. The public repository currently contains only a license. [paper](https://arxiv.org/abs/2604.26508) | [nominal code repository](https://github.com/open-ep/ProSemComVLM)

**System A (Cheng Yuan, Zhening Liu, Jiashu Lv, Jiawei Shao, Yufei Jiang, Jun Zhang, and Xuelong Li):** Used the direct split with task-aware feature coding in **Task-Oriented Feature Compression for Multimodal Understanding via Device-Edge Co-Inference (TOFC)**. The device runs the complete pretrained SigLIP encoder for LLaVA-OneVision-7B, clusters and merges its output to 1.1-4.4% of the original feature count, and entropy-codes those merged features before projection. The server entropy-decodes features rather than pixels, then runs the target projector and LLM. On seven VQA benchmarks, the current paper reports up to 52% less transmitted data and 63% lower system latency than the learned image codec ELIC at matched task performance. Its LLaVA-1.5 extension reports 48-70% lower transmission than four image codecs at matched target performance and 68-85% lower E2E latency than server-only inference while retaining 96.5% of the backbone benchmark score. Main measurements use a Jetson AGX Orin and RTX 4090 server; about 300M trainable parameters require 8.1 hours on eight RTX 4090 GPUs. The official repository is an archived research snapshot with hard-coded paths and an intermediate checkpoint, and explicitly omits the two-device cooperative-inference path, so it is not an end-to-end service. [paper](https://arxiv.org/abs/2503.12926) | [partial code](https://github.com/asdLeaving/TOFC)

**System B (Xu Liu, Guikun Chen, Zihao Yan, Kanzhi Wu, and Wenguan Wang):** Published the latest qualifying deployment proposal found, **Co-VStream: Edge-Cloud Collaboration for Understanding of Long Video Streams**, in June 2026. The edge runs VideoLLaMA3-7B's complete 0.4B vision encoder and condenses each 16-frame feature window to two feature groups. It also runs a separate Qwen3-VL-2B captioner on selected keyframes. The cloud feeds the condensed visual context directly to VideoLLaMA3-7B and augments it with an entity graph built from captions. On LVBench, communication falls 87.59%, accuracy is 38.93% versus 39.23% cloud-only, and response latency is 2.99 s versus 4.08 s. All main experiments use an A40; edge feasibility is stress-tested on RTX 3090 and RTX 2080 GPUs, so its physical edge evidence is weaker than the i.MX95 testbed above. Co-VStream literally satisfies the complete-encoder/server-LLM definition, but it is not a clean test of embeddings alone because captions and graph memory materially improve accuracy. No public code was identified. [paper](https://arxiv.org/abs/2606.22804)

**Paper 3 (Xiao Liu, Lijun Zhang, Deepak Ganesan, and Hui Guan):** Used a closely related intra-encoder split in **Aligned Vector Quantization for Edge-Cloud Collaborative Vision-Language Models**. A Jetson AGX Xavier computes LLaVA-1.5's patch embedding and part of its first CLIP transformer block. A learned vector quantizer sends codebook indices to an A100 server, which runs the remaining encoder blocks, projector, and Vicuna-7B LLM. The original paper reported 0.845 KiB, 1,365x feature compression, 96.8% fewer bytes than JPEG quality-90, and a 2-15x vision-pipeline speedup. The April 2026 revision states: "The residual links are not taken into consideration when computing the transmission. All results about the compressed data size and transmission latency would be affected." No corrected totals are reported, so those network claims are historical, invalidated estimates rather than usable results. Its accuracy results, between 2.23 percentage points lower and 1.6 points higher than unchanged LLaVA across benchmark groups, were not explicitly withdrawn. It requires one-epoch task-aware training of the codebook, adapter layers, and low-rank adaptation weights. No public source implementation was found. [paper and revision notice](https://arxiv.org/abs/2411.05961)

### Serving and resource disaggregation

**Paper 4 (Haoran Qiu, Anish Biswas, Zihan Zhao, Jayashree Mohan, Alind Khare, Esha Choukse, Inigo Goiri, Zeyu Zhang, Haiying Shen, Chetan Bansal, Ramachandran Ramjee, and Rodrigo Fonseca):** Applied the same model boundary inside a datacenter in **ModServe**. "Image Instances" are GPU processes that preprocess and encode images; "Text Instances" are separate GPU processes that run language-model prefill and decoding. A request containing ten image tiles can send different tiles to different Image Instances because those tiles do not depend on each other during encoding. The resulting tokens are gathered by one Text Instance. ModServe also scales the two process pools independently when image traffic and text traffic grow at different rates. On 16 servers containing 128 A100 GPUs and replayed Azure multimodal traces, it supports 3.3-5.5 times the request arrival rate of monolithic serving while staying within the same tail-latency targets, translating to 25-41.3% reported cost savings. With remote direct memory access, the 99th-percentile internal token-transfer delay is 5 ms; ordinary datacenter Ethernet reaches 180 ms. These are cluster-network results, not mobile Internet results. Raw images still enter trusted Image Instances, so ModServe demonstrates scheduling and scaling benefits rather than client privacy. The paper describes a 5,000-line Python prototype, but no public ModServe source repository was found; only the evaluation trace is public. [paper](https://arxiv.org/abs/2502.00937) | [Azure trace](https://github.com/Azure/AzurePublicDataset)

**Platform 1 (vLLM and llm-d contributors):** vLLM implements two separate capabilities. First, its OpenAI-compatible Chat Completions API accepts base64-encoded `image_embeds` when the server starts with `--enable-mm-embeds`. Model-specific metadata may also be required: for example, Qwen2-VL needs its image grid dimensions. vLLM warns that an incorrect embedding shape may crash the engine and says to enable the flag only for trusted users. Second, vLLM implements **disaggregated encoding**, meaning a trusted encoder process is separate from language-model prefill and decode and transfers cached outputs through an Encoder Cache Connector. llm-d packages this second capability into experimental `E/PD` and `E/P/D` cluster recipes using CPU shared memory, ZeroMQ control, and NIXL transfer. llm-d's documented external request still contains ordinary media; it does not demonstrate raw embedding ingress even though its underlying vLLM engine has that separate feature. [vLLM multimodal inputs](https://docs.vllm.ai/en/latest/features/multimodal_inputs/) | [vLLM disaggregated encoder](https://docs.vllm.ai/en/latest/features/disagg_encoder/) | [llm-d recipe](https://github.com/llm-d/llm-d/tree/main/guides/multimodal-serving/e-disaggregation)

**Platform 2 (NVIDIA TensorRT-LLM contributors):** Supports two mechanisms. External-embedding support, merged July 30, 2025, lets the Python LLM API and current `trtllm-serve` path accept serialized image embedding tensors for supported models. The standalone multimodal encoder, merged August 20, 2025, instead runs only the ViT and returns an `mm_embedding_handle`, an opaque identifier for a tensor in shared memory. That handle is for cooperating server processes and is not a portable client payload. A cross-device product can build on the direct tensor path, but still needs authentication, strict shape and size validation, model-version negotiation, and a safer stable wire schema. [external embeddings PR #6263](https://github.com/NVIDIA/TensorRT-LLM/pull/6263) | [standalone encoder PR #6743](https://github.com/NVIDIA/TensorRT-LLM/pull/6743)

**Platform 3 (NVIDIA Dynamo contributors):** Supports encoder/prefill/decode disaggregation across multiple backends. Its vLLM backend can process embedding and base64 inputs when multimodal support is explicitly enabled. Its documented TensorRT-LLM precomputed-embedding route accepts allowlisted server-local `.safetensors` paths and moves tensors internally through NIXL; that route is appropriate for a controlled cluster or prior upload service, not arbitrary Internet tensor bytes. [encoder disaggregation](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/features/multimodal/encoder-disaggregation.md) | [vLLM multimodal backend](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/features/multimodal/multimodal-vllm.md) | [TensorRT-LLM multimodal backend](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/features/multimodal/multimodal-trtllm.md)

### Token and representation compression

**Paper 5 (Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia):** Reduced the payload-ready output of a vision encoder in **VisionZip**. It reads the encoder's internal attention scores to identify "dominant" tokens that other tokens attend to strongly. It keeps those tokens and groups the remaining tokens by similarity, replacing each group with an average. On LLaVA-1.5 it reduces 576 visual tokens to 192, 128, or 64. Its "98.5%, 97.6%, and 94.0% performance" values are normalized averages over 11 VQA and multimodal benchmarks, where unchanged 576-token LLaVA is defined as 100%; they are not literal accuracy on one dataset. The training-free 64-token setting therefore loses 6% of that normalized average. Fine-tuning only the projector for 30 minutes on eight A800 GPUs raises it to 95.2%. On LLaVA-NeXT, reducing the tokens consumed by the LLM lowers prefill time by as much as eight times. It does not avoid running the full vision encoder. In the LLaVA-1.5 example, 64 projected FP16 tokens still occupy `64 * 4096 * 2 = 524,288` bytes, or 512 KiB, so pruning alone does not beat the brief's image-size estimate. [paper](https://arxiv.org/abs/2412.04467) | [code](https://github.com/dvlab-research/VisionZip)

**Paper 6 (Kevin Y. Li, Sachin Goyal, Joao D. Semedo, and J. Zico Kolter):** Studied a fixed LLM-compute budget in **Inference Optimal VLMs Need Fewer Visual Tokens and More Parameters**. If two configurations cost about the same, their fitted scaling law often favors a larger language model receiving 1-16 visual tokens over a smaller language model receiving hundreds. This is a compute-allocation result, not a claim that one token exactly preserves the 576-token model. Their Query-based Convolutional Cross-attention compressor, abbreviated **QueCC**, uses the user's text question to decide what visual information to preserve. With one visual token and a 7B Vicuna language model, QueCC scores 53.5 on GQA versus 62.0 for full LLaVA, 59.4 versus 64.3 on MMBench, and 67.3 versus 78.5 on VQAv2. It is still better than competing one-token compressors on six of eight evaluated datasets. OCR and document tasks favor more visual tokens because one summary vector cannot retain all visible characters. QueCC must be trained and needs an LLM-derived representation of the question; unless that representation is cached for a fixed prompt, a standalone image-only client encoder cannot run it independently. [paper](https://arxiv.org/abs/2411.03312) | [code](https://github.com/locuslab/llava-token-compression)

**Paper 7 (Pavan Kumar Anasosalu Vasu et al., Apple):** Redesigned the encoder rather than pruning its output in **FastVLM**. Its FastViTHD encoder progressively reduces image resolution inside a convolution/transformer hierarchy, so a 256x256 input produces only 16 visual tokens. In models trained under the same LLaVA-1.5 recipe, the authors compare total vision encoding plus LLM prefill on an M1 MacBook Pro and report up to 3.2 times lower time to first token at similar average benchmark quality than common ViT encoders. The separate 85-times result compares complete systems: FastVLM and LLaVA-OneVision both use a 0.5B language model and a 1152x1152 input, but they have different encoders, token counts, and training. It therefore should not be read as an isolated 85-times encoder speedup transferable to another VLM. FastVLM can reduce both client work and wire tokens, but adopting it means using a VLM trained with FastViTHD rather than splitting an arbitrary existing checkpoint. [paper](https://arxiv.org/abs/2412.13303) | [code and models](https://github.com/apple/ml-fastvlm)

**System 1 (Bo Li and contributors):** **Sema** moves a discrete visual tokenizer to the client and sends NVIDIA Cosmos codebook indices, roughly 3-8 KB, over WebSocket. The server reconstructs the image and invokes an ordinary Qwen2.5-VL service. Its "hybrid" mode also sends structured text taken from the browser's document structure or local OCR, which can directly reveal button labels and form text. On 30 generated browser screenshots at 5 Mbps uplink and 50 ms round-trip time, hybrid mode uses 6,548 average uplink bytes versus 9,738 for WebP and reaches 100% on 21 navigation and 9 text tasks. Cosmos tokens alone score 0% on both, while raw WebP scores 100% navigation and 0% text. The structured-text side channel is therefore essential and makes this an unfair image-codec-only comparison. The sample is also too small and synthetic for a general quality claim. It is Apache-2.0 and runnable, but it is not the target architecture because the server reconstructs pixels and reruns a separate VLM vision encoder. [code](https://github.com/bojieli/Sema)

### Privacy and security results

**Paper 8 (Kedong Xiu and Sai Qian Zhang):** Demonstrated that conventional encoder features still leak high-level content in **CapRecover**. The attacker knows the victim encoder architecture, collects separate images with captions, and trains a small adapter that maps stolen encoder features directly to a frozen language model. The attacker does not need to recreate the pixels first. On COCO2017, captions recovered from CLIP ViT-L/16 features reach ROUGE-L 0.53; ROUGE-L measures overlap in the longest matching word sequence between recovered and reference captions, where higher is better but paraphrases may score poorly. The paper also embeds both captions and counts 84.38% of recovered examples above cosine similarity 0.7, its chosen semantic-match threshold. On CIFAR-10, where an image belongs to one of ten simple classes such as airplane, cat, or truck, the attacker predicts the correct class first 92.71% of the time. This does not show immediate pixel-perfect reconstruction, but it directly disproves the claim that encoder outputs hide objects and scene meaning. [paper](https://arxiv.org/abs/2507.22828)

**Paper 9 (Chenyu Zhou, Qiliang Jiang, Shuning Wu, and Xu Zhou):** Compared encoder-free and encoder-based models in **The Vision Encoder as a Privacy Boundary**. Here, **inversion** means training a decoder that maps intercepted visual tensors back toward a recognizable image. Gemma 4 Unified and Fuyu map relatively raw spatial patches into the language stream without a deep semantic vision encoder. Their tensors reconstructed exact five-character access codes in 42 of 48 and 46 of 48 held-out pages. Under the same attack pipeline, Qwen3-VL, InternVL, and LLaVA-1.5 encoder outputs recover 0 of 48 exact codes, although they still localize the region containing text. Gemma's early attention memory, its KV cache, is also invertible. Adding Gaussian noise equal to 10% of token standard deviation still recovers 18 of 24 codes, and converting each feature to 3 bits still recovers 19 of 24. Reducing the spatial grid from 12x22 to 6x11 recovers 0 of 24 because it removes character-level spatial samples. This is evidence about exact pixel/text detail in the tested architectures, not proof that encoder-based features are private: CapRecover shows that their high-level semantics remain recoverable. [paper](https://arxiv.org/abs/2606.14783)

**Paper 10 (Zikai Zhang, Rui Hu, Olivera Kotevska, and Jiahao Xu):** Exposed an integrity risk in **Vision Token Manipulation Attacks on Cloud-Edge Inference of Large Vision-Language Models**. A man-in-the-middle attacker can read and alter traffic but cannot inspect either model. The attack preserves tensor shape so ordinary shape checks pass, then reverses the sign of selected vectors, changing `v` to `-v` without changing its length. With only 10% of visual-token rows modified, optimized selection reduces Qwen2.5-VL-72B MMBench accuracy from 88.39% to 0.08%, an 88.31 percentage-point drop. InternVL3.5 is more robust in many settings, showing the result is architecture-dependent. Correctly authenticated Transport Layer Security, TLS, prevents an outside network attacker from silently changing bytes inside one TLS connection. Application-level signatures or message authentication are still useful when TLS terminates at proxies, tensors enter queues or shared storage, or end-to-end integrity must survive several internal services. Numeric shape and norm checks improve stability but cannot prove a plausible-looking tensor came from a real image. The public repository currently contains only an appendix PDF. [paper](https://arxiv.org/abs/2607.02819) | [artifact](https://github.com/superkevingit/Vision-Token-Manipulation-Attack)

### Adjacent approaches worth tracking

**Paper 11 (Yuanyuan Jia, Shunpu Tang, and Qianqian Yang):** Used **speculative decoding** in **CoVSpec**. Speculative decoding lets a small model propose several answer tokens, then asks a large model to verify all proposals in one operation. A complete InternVL2.5-4B draft VLM runs on a MacBook Pro M5, while an INT8 InternVL2.5-78B verifier runs on two RTX 4090s. The draft model keeps 64 of 768 visual tokens. Across VQAv2, MMMU, and MMBench, it reports 1.63-2.21 times the generated-token speed of large-model-only inference. Communication is 16.49-21.23 MB for a 1,024-token generation, versus 534.51-599.23 MB for ordinary device-edge speculative decoding that repeatedly exchanges full probability arrays. This is still far larger than one image upload and solves autoregressive synchronization rather than raw-image privacy; the server verifier needs its own full visual context. [paper](https://arxiv.org/abs/2605.02218)

**Paper 12 (Chen Qian et al.):** Used delayed large-VLM answers as context for a real-time local small VLM in **edgeVLM**. It uploads selected video keyframes, runs a local Qwen2.5-VL-3B, and reuses delayed Qwen2.5-VL-72B results to improve future frames. This is useful when cloud jitter makes a split pipeline miss frame deadlines, but it transmits images and runs two complete VLMs. Code was promised but no repository URL was found. [paper](https://arxiv.org/abs/2508.12638)

**Paper 13 (Soochang Song and Yongjune Kim):** **Collaborative Edge-to-Server Inference for Vision-Language Models** first sends a low-resolution global image. The server measures **output entropy**, a numerical measure of how spread out or uncertain the model's next-token probabilities are. If uncertainty is high, the server uses its attention map to select a region of interest, meaning a bounding-box crop likely to contain the missing detail, and asks the client for that high-quality crop. This can avoid uploading a full-resolution image when a global view is sufficient, but the server receives pixels, runs the whole VLM, and may add a second network round trip. It is an alternative for clients too weak to run any vision encoder, not the target split. [paper](https://arxiv.org/abs/2512.16349)

**Paper 14 (INAR-VL authors):** Routes complete requests between two INT8 edge VLMs and two FP16 cloud VLMs in **INAR-VL**. On an RTX 4060 edge and RTX PRO 6000 cloud setup, it keeps 36% of requests local, reduces latency by 24% and energy by 26%, and preserves 97% of cloud accuracy. It is an important alternative if the product can accept local-model answers for easy requests, but it is not one partitioned model. [paper](https://arxiv.org/abs/2605.18853)

### Timeline

| Period | Approach | Representative work | Main lesson |
|---|---|---|---|
| Earlier split computing | Split CNNs/transformers at intermediate layers | General split-DNN literature | Intermediate activations can cost more than inputs and can be inverted. |
| 2024 | Task-aware activation compression inside a VLM encoder | AlignedVQ | Discrete indices looked promising, but its published wire-size and latency accounting was later declared incomplete. |
| 2024-2025 | Token pruning/merging before or inside the LLM | FastV, PruMerge, VisionZip, QueCC | 80-97% token reduction can preserve visual reasoning; OCR/detail tasks degrade sooner. |
| 2025 | Whole encoder on edge, decoder on server | Distributed VLMs | Pipeline overlap can improve throughput even without compression. |
| 2025 | Whole encoder plus feature compression on device | TOFC | Pre-projector feature merging and entropy coding can beat image codecs at matched task quality. |
| 2025 | Encoder pools separated from decoder pools | ModServe, TensorRT-LLM | Independent scaling is valuable in production, but token transfer needs fast networking. |
| 2025-2026 | Adaptive pixel/ROI transfer | VaVLM, entropy-aware retransmission | Useful when clients cannot run the encoder or fine detail is needed only sometimes. |
| 2026 | Progressive latent transfer | Progressive Semantic Communication | A learned latent can adapt payload to bandwidth/quality, demonstrated on embedded hardware. |
| 2026 | Continuous feature and caption streaming | Co-VStream | Feature condensation can support long-video memory, but its caption/graph side channel makes attribution difficult. |
| 2025-2026 | Privacy and integrity attacks | CapRecover, visual-token side channels, VTM-Attack | Embeddings are sensitive data and the transport boundary must be secured. |

## 2. What is the latest attempt?

**Co-VStream**, first posted June 22, 2026 and revised June 25, is the latest qualifying deployment proposal found through the July 29 cutoff. It runs the complete VideoLLaMA3-7B vision encoder at the edge and sends condensed features to the same model family's cloud reasoner without reconstructing pixels. Its answer quality also depends on edge-generated captions and a cloud entity graph, and its edge hardware evaluation uses desktop RTX GPUs rather than a phone-class device.

**Progressive Semantic Communication**, posted April 29, 2026, is therefore the latest cleaner physical realization found of the specific proposal in the brief. It demonstrates the complete SmolVLM vision encoder on an NXP i.MX95, progressive latent transfer over a throttled link, and the matching LLM on a separate RTX 2080 Super server. The July 2026 Vision Token Manipulation paper uses a similar split but is a security attack study, not a newer deployment design.

## 3. Is there an inference platform that offers it?

The word **platform** has two relevant meanings. A hosted model API supplies a provider-owned endpoint and fixed request schema. An infrastructure platform runs code or containers supplied by the customer. No checked provider offers precomputed VLM visual tensors through its standard hosted-model API, but Modal and Baseten can run a custom implementation.

### Hosted inference providers

| Provider | Turnkey precomputed VLM embedding API? | Can host a custom implementation? | Finding |
|---|---|---|---|
| [Modal](https://modal.com/blog/how-to-deploy-vllm) | No | Yes | Modal documents custom Python endpoints and deploying vLLM. A deployment can start vLLM with `--enable-mm-embeds` and place a validation endpoint in front of it. This is project code running on Modal, not a Modal-provided embedding contract. |
| [Baseten](https://docs.baseten.co/development/model/custom-server) | No | Yes | Baseten supports arbitrary Truss `/predict` handlers and custom containers such as vLLM. Its standard vision Model API accepts `image_url` inputs and runs vision processing within the service. |
| [Fireworks](https://fireworksai.mintlify.app/guides/querying-vision-language-models) | No | Not through the documented VLM API | Its documented VLM request accepts image URLs or base64-encoded image pixels. Its embeddings API is for retrieval vectors, not injecting visual tokens into a VLM. |
| [Together AI](https://docs.together.ai/docs/inference/embeddings/embeddings) | No | No documented direct route found | Its embeddings API creates text/search vectors. That is different from supplying the target VLM's intermediate visual-token sequence. |
| [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/en/index) | No | A separately managed custom endpoint could | The shared provider interface exposes normal VLM chat and feature-extraction tasks, not a standard model-specific visual-tensor contract. |

An API named **embeddings** normally performs `text or image -> retrieval vector`. This project needs the reverse serving boundary: `client vision encoder -> model-specific visual-token sequence -> matching projector/LLM`. Those interfaces are not interchangeable.

The most direct experimental deployment is a custom Modal or Baseten endpoint that authenticates the caller, validates and dequantizes the tensor, and then calls a private vLLM process. Baseten supplies more production-oriented custom-container and model-deployment machinery; Modal supplies a flexible way to define the Python/container boundary. Neither removes the need to implement serialization, model revision matching, shape limits, safety checks, and decompression.

### Serving engines and cluster platforms

| Platform | What exists | Can an external client send its own embeddings? | Maturity |
|---|---|---|---|
| vLLM | Direct `image_embeds` input plus separate encoder/PD instances with EC Connector | Yes, base64 tensors through Chat Completions with `--enable-mm-embeds`; trusted users only | Input API exists; disaggregation is developer preview |
| llm-d | Kubernetes E/PD and E/P/D deployments for Qwen3-VL-32B | No; client sends ordinary media and cluster encoder workers produce embeddings | Experimental, active development |
| TensorRT-LLM | External embedding input and a separate standalone multimodal encoder | Yes through the external-tensor path; the encoder-only path returns a non-portable shared-memory handle | Merged open-source features |
| NVIDIA Dynamo | Multimodal E/PD and E/P/D across serving backends | vLLM backend can accept embeddings; documented TensorRT-LLM path uses server-local `.safetensors` | Active open-source platform |
| ModServe | Research prototype with image/text pools and RDMA | No public service or source found | Research prototype |
| Hugging Face Transformers | Model internals expose `pixel_values`, vision towers, projectors, and `inputs_embeds`-like paths | Possible in a custom server, but not a standard hosted VLM contract | Building blocks, not a service |
| Hosted OpenAI-compatible APIs | Normal image URL/base64 input | No checked standard hosted endpoint accepts model-specific visual tensors | Not offered as a turnkey product |

Likely engineering obstacles for a hosted embedding-input API, inferred from the model contracts and attacks above rather than stated provider policy:

- Embeddings are model-, revision-, processor-, dtype-, and projector-specific.
- Untrusted tensors can bypass input validation and image safety checks.
- Shape-compatible malicious tensors can corrupt behavior.
- Providers cannot easily meter, cache, batch, or reproduce requests without an explicit contract.
- The API would need versioned tensor schemas, integrity validation, limits, and compatibility negotiation.

## 4. Are there open-source implementations?

"Runnable" below means the repository contains executable source and instructions for the relevant path. It does not mean this report reproduced the system. Repository contents can change after the check date, so the links and 2026-07-29 status are part of each assessment.

| Project | Relevant implementation | Status as checked |
|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | Direct `image_embeds` input, `vllm/distributed/ec_transfer`, E->PD and E->P->D examples | Direct input source exists; disaggregation examples are developer preview |
| [llm-d](https://github.com/llm-d/llm-d) | Kubernetes encode-disaggregation recipe, EC/ECCPU connectors | Deployment recipe and source exist; explicitly experimental |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Direct external tensors, standalone encoder, embedding handles | Both paths merged with tests; handle path is for shared-memory server processes |
| [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) | Multimodal routing and encoder/prefill/decode disaggregation | Public source and deployment documentation; backend-specific input restrictions |
| [TOFC](https://github.com/asdLeaving/TOFC) | Feature merging, learned entropy model, intermediate checkpoint | Partial archived research code; cross-device transport is explicitly absent |
| [VisionZip](https://github.com/dvlab-research/VisionZip) | Training-free token selection/merging | Public code |
| [LLaVA token compression](https://github.com/locuslab/llava-token-compression) | QueCC and low-token experiments | Public code |
| [FastVLM](https://github.com/apple/ml-fastvlm) | Efficient encoder and complete VLM checkpoints | Public code and models |
| [Sema](https://github.com/bojieli/Sema) | Client discrete tokenization, WebSocket transport, server reconstruction | Working reference, but a different architecture |
| [ProSemComVLM](https://github.com/open-ep/ProSemComVLM) | Paper-linked placeholder for a promised release | Empty except GPL license; not usable yet |
| [VTM-Attack](https://github.com/superkevingit/Vision-Token-Manipulation-Attack) | Claimed attack artifact | Appendix PDF only; not usable code yet |
| Distributed VLMs | Exact split system | No public code found |
| Co-VStream | Complete encoder, feature condensation, captions, and cloud reasoning | No public code found |
| ModServe | Encoder/text pool serving | No public source found; evaluation trace is public |

## Encoder size and per-frame compute

### Method and caveats

In this section, one **frame** means one still RGB image presented to the VLM. A high-resolution frame may be divided into several **tiles**, meaning resized rectangular images that the encoder processes independently. Each tile is divided into patches. The number of encoder patches can be larger than the number of visual tokens finally sent to the LLM because pooling or patch merging combines neighboring encoder outputs.

Published papers rarely report encoder FLOPs consistently, especially for dynamic-resolution and tiled models. The table therefore estimates a dense Vision Transformer using `N` for input patch tokens, `d` for encoder hidden width, `d_mlp` for the wider feed-forward layer, and `L` for transformer layers:

- Published parameter counts and architecture dimensions where available.
- One multiply plus one add = **2 FLOPs**.
- Dense ViT estimate per layer:

```text
per-layer FLOPs ~= 2 * [4*N*d^2 + 2*N^2*d + 2*N*d*d_mlp]
encoder FLOPs ~= patch projection + L * per-layer FLOPs
```

The `4*N*d^2` term covers query, key, value, and output projections in attention. The `2*N^2*d` term covers comparing every token with every other token and combining attention values. The `2*N*d*d_mlp` term covers the two feed-forward matrix multiplications. The outside factor of two counts one multiply and one add separately.

- Patch projection is included. Resize/normalization, LayerNorm, bias and activation functions, pooling, multimodal projector, pruning/compression module, serialization, network transfer, and LLM work are excluded unless noted.
- Tiled images are encoded as independent tiles. Dynamic models are shown at an explicit token budget.
- These are architecture estimates for comparing orders of magnitude, not measured device throughput. Kernel efficiency, sparsity, quantization, local/window attention, and memory traffic change wall-clock latency. A model requiring fewer FLOPs can still be slower on hardware with a poor implementation.

### Comparison table

#### Current open-weight candidates

This focused table covers the current model families requested for implementation planning. **Vision parameters** include the vision tower and its model-specific output merger or projection where applicable, not the language model. Weight sizes are ideal raw sizes before file-format metadata or runtime buffers. FP16 and BF16 both require two bytes per parameter; INT8 requires approximately one byte per parameter plus quantization metadata.

For Qwen3.6 and Holo3.1, one representative frame is fixed at 512x512: 1,024 vision-transformer tokens before 2x2 spatial merging and 256 visual tokens afterward. Gemma 4 uses its default 280-soft-token image setting; its conventional encoder processes about nine patch tokens per output token. The FLOP values are architecture estimates under those assumptions, not measured latency.

| VLM | Client vision path | Vision parameters | FP16/BF16 weights | INT8 weights | Approx. work per frame |
|---|---|---:|---:|---:|---:|
| Gemma 4 12B Unified | Patch merging and direct projection; no transformer vision tower | 49.9M complete image embedder; ~35M main path reported in paper | ~99.8 MB | ~49.9 MB | ~23.1 GFLOPs for the two dense projections |
| Holo3.1-0.8B | 12-layer Qwen3.5-derived vision transformer | ~99M | ~198 MB | ~99 MB | ~222 GFLOPs |
| Holo3.1-4B | 24-layer Qwen3.5-derived vision transformer | ~331M | ~662 MB | ~331 MB | ~739 GFLOPs |
| Qwen3.6-35B-A3B | 27-layer Qwen vision transformer | ~444M | ~888 MB | ~444 MB | ~0.99 TFLOP |
| Holo3.1-35B-A3B | 27-layer Qwen3.5-derived vision transformer | ~444M | ~888 MB | ~444 MB | ~0.99 TFLOP |
| Holo3.1-9B | 27-layer Qwen3.5-derived vision transformer | ~453M | ~906 MB | ~453 MB | ~1.00 TFLOP |
| Qwen3.6-27B | 27-layer Qwen vision transformer | ~458M | ~916 MB | ~458 MB | ~1.00 TFLOP |
| Gemma 4 26B-A4B | 27-layer Gemma vision transformer | ~550M | ~1.10 GB | ~550 MB | ~2.87 TFLOPs |
| Gemma 4 31B | Same 27-layer Gemma vision transformer | ~550M | ~1.10 GB | ~550 MB | ~2.87 TFLOPs |

The actual Gemma mixture-of-experts checkpoint is **Gemma 4 26B-A4B**, not 27B-A3B. Qwen3.6 and Holo3.1 provide 35B-A3B checkpoints. The language model's sparse active-parameter count does not make its vision encoder sparse: Qwen3.6-35B-A3B and Holo3.1-35B-A3B still run approximately 444M vision parameters for every image.

Holo3.1 is not uniformly lightweight. Its 0.8B checkpoint has the smallest conventional deep encoder in this focused set, but Holo3.1-9B and Holo3.1-35B-A3B have roughly the same client vision cost as Qwen3.6. Holo is specialized for screenshots, interface grounding, and computer-use actions, so its smaller checkpoints are most relevant if Step 2 chooses browser or desktop interaction rather than general visual conversation.

Gemma 4 12B Unified is much smaller in vision-side parameters, but it is not a conventional semantic encoder. The paper-level estimate of about 35M covers its main patch and positional path; exporting the complete Hugging Face image embedder measured 49,922,304 parameters after including the final multimodal projection. At 280 output tokens of width 3,840, its ideal uncompressed BF16 tensor is about 2.05 MiB. Its low projection FLOPs also exclude the language-model work that subsequently processes those visual tokens. Most importantly for this brief, the reviewed privacy attack recovered exact access codes much more often from this shallow, spatially local representation than from the tested deep encoder outputs.

#### Broader reference models

| VLM / vision path | Encoder parameters | Frame assumption | Approx. encoder work | Candidate split tensor | Projector location for stated tensor | Ideal FP16 bytes |
|---|---:|---|---:|---|---|---:|
| SmolVLM-256M | ~86M SigLIP-B/16, estimated from config | 512x512, 1,024 image patches plus special token | ~214 GFLOPs, estimated | 64 x 576 | Included in the stated output path | ~72 KiB |
| LLaVA-1.5 7B/13B | ~304M CLIP ViT-L/14, published model size | 336x336, 576 patches plus class token | ~382 GFLOPs, estimated | 576 x 4,096 | Client-side for this post-projector example | 4.50 MiB |
| Qwen2.5-VL-7B | ~675M vision transformer, reported family size | 1,024 raw patch tokens, 256 after 2x2 merge | ~1.18 TFLOPs, estimated dense upper approximation | 256 x 3,584 | Client-side for this example | 1.75 MiB |
| LLaVA-OneVision 7B/72B | 400M SigLIP, reported by ModServe | One 384x384 tile, 729 tokens | ~666 GFLOPs/tile, estimated | 729 x decoder width | Model dependent | Not stated |
| LLaVA-OneVision, ModServe example | Same 400M SigLIP | One 896x896 image represented by 10 independent tiles | ~6.66 TFLOPs, estimated as 10 tile runs | 7,290 x decoder width | Model dependent | Tens of MiB, width dependent |
| Llama 3.2 Vision 11B/90B | 630M ViT-H/14, reported by ModServe | One 560x560 tile, 1,600 patches plus special token | ~2.44 TFLOPs/tile, estimated | 1,601 cross-attention vectors | Cross-attention architecture; not directly comparable to LLaVA projector output | Not stated |
| Llama 3.2 Vision, ModServe example | Same 630M ViT-H/14 | One 896x896 image represented by 4 independent tiles | ~9.75 TFLOPs, estimated as 4 tile runs | 6,404 cross-attention vectors | Same as above | Not stated |
| InternVL2.5-26B / NVLM-D-72B | 6B InternViT, reported by ModServe | One 448x448 tile, 1,024 input patches | ~12.7 TFLOPs/tile, estimated | 256 projected vectors/tile | Included in stated token count | Width dependent |
| InternVL/NVLM, ModServe example | Same 6B InternViT | One 896x896 image represented by 5 independent tiles | ~63.6 TFLOPs, estimated as 5 tile runs | 1,280 projected vectors | Included | Width dependent |
| Gemma 4 E2B/E4B | 150M ViT, reported | 70-1,120 final soft-token budget; encoder processes about 9 patches per final token before 3x3 pooling | ~163 GFLOPs-7.29 TFLOPs, estimated | 70-1,120 soft tokens | Model dependent | Width dependent |
| Gemma 4 26B-A4B/31B | 550M ViT, reported | Same token budgets | ~568 GFLOPs-20.9 TFLOPs, estimated | 70-1,120 soft tokens | Model dependent | Width dependent |
| Gemma 4 12B Unified | No transformer vision tower; reported ~35M main patch projection | 70-1,120 merged 48x48 RGB patches | ~4.9-78.4 GFLOPs for the reported main projection only | 70-1,120 x 3,840 | Direct projection path | ~0.51-8.20 MiB |

Sources for the tables: [Qwen3.6-27B config](https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json), [Qwen3.6-35B-A3B config](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json), [Holo3.1 model collection](https://huggingface.co/Hcompany), [Gemma 4 12B Unified config](https://huggingface.co/google/gemma-4-12B-it/blob/main/config.json), [Gemma 4 26B-A4B config](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/main/config.json), [Gemma 4 31B config](https://huggingface.co/google/gemma-4-31B-it/blob/main/config.json), [LLaVA config](https://huggingface.co/llava-hf/llava-1.5-7b-hf/blob/main/config.json), [Qwen2.5-VL config](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json), [SmolVLM config](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/blob/main/config.json), [ModServe Table 1](https://arxiv.org/abs/2502.00937), and [Gemma 4 report](https://arxiv.org/abs/2607.02770).

### Gemma 4 Unified clarification

Gemma 4 12B Unified is the model mentioned in the brief. It is not literally free of visual work. It removes the 150M/550M transformer vision tower and performs:

1. 16x16 patchification.
2. 3x3 patch merging into 48x48x3 = 6,912-value patches.
3. LayerNorm, a large dense patch projection, and LayerNorm.
4. Factorized 2D positional embeddings.
5. Final normalization/projection into the 3,840-wide LLM space.

The report describes the main patch and positional path as about 35M parameters, versus 150M or 550M for the transformer vision towers. The complete exported Hugging Face image module has 49,922,304 parameters because it also contains the final 3,840-to-3,840 multimodal projection. It therefore performs substantially less vision-side arithmetic in the estimate above, but preserves a dense, pixel-local spatial grid. In the June 2026 attack, its tokens recovered 42 of 48 held-out access codes while three tested encoder-based controls recovered none. This makes it attractive for client compute but riskier for exact text leakage under that attack model.

## Bandwidth analysis

This section uses KiB for 1,024 bytes and MiB for 1,048,576 bytes. The source brief uses the approximate range 50-70 KB; the threshold table rounds that to 50-70 KiB so every listed token count actually fits below its stated bound. Here, **beats the image** means only that the tensor has fewer ideal uplink bytes than that estimate. It does not automatically mean lower latency, client energy, cost, or better task quality.

For `N` visual tokens, each containing `d` numeric values stored with `b` bytes per value:

```text
payload_bytes = N * d * b + protocol overhead
```

Ignoring protocol and quantization metadata, the largest token count whose raw values fit below the image estimate is:

| Embedding width | Encoding | Fits within 50 KiB | Fits within 70 KiB |
|---:|---:|---:|---:|
| 4,096 | FP16 (2 B) | 6 tokens | 8 tokens |
| 4,096 | INT8 (1 B) | 12 tokens | 17 tokens |
| 4,096 | 4-bit | 25 tokens | 35 tokens |
| 3,584 | FP16 | 7 tokens | 10 tokens |
| 1,024, before projector | FP16 | 25 tokens | 35 tokens |
| 1,024, before projector | INT8 | 50 tokens | 70 tokens |

Implications:

- These are ideal lower bounds. INT8 needs at least quantization scales, and 4-bit data needs scales, packing alignment, and often zero points. Every request also needs a tensor header, model metadata, authentication tag, and possibly padding.
- An 80% reduction from 576 to 115 tokens is not enough at normal decoder widths. At width 4,096 in FP16, `115 * 4096 * 2 = 942,080` bytes, about 920 KiB.
- VisionZip at 64 x 4,096 FP16 is about 512 KiB, still much larger than a 50-70 KB image.
- Moving LLaVA's projector to the server changes each token from width 4,096 to 1,024, cutting raw bytes by four. It also makes the server responsible for the exact projector weights and means the transmitted object is an encoder output rather than an LLM-ready embedding.
- SmolVLM's 64 x 576 FP16 representation is roughly 72 KiB, already close to the image sizes in the brief before compression.
- AlignedVQ's original 0.845 KiB estimate counted packed 12-bit codebook indices but omitted residual links. Its authors later said all compressed-size and transmission-latency results were affected, so it is not a valid payload baseline.
- Base64 expands binary by about one third. A JSON list of decimal floats adds variable text characters, commas, and parsing overhead and can be several times larger than binary. A valid benchmark must compare encrypted bytes actually written to the socket for the same source images and an agreed maximum quality loss.

## Privacy and threat model

"Private" depends on who is being defended against:

- An **honest-but-curious server** follows the protocol but analyzes received tensors to infer more about the image. TLS does not help against this server because the server decrypts the request normally. CapRecover is relevant here.
- A **network interceptor** observes or changes traffic between client and server. Correctly authenticated TLS protects against this attacker while data stays inside the connection. The token-manipulation paper shows what can happen if integrity is absent or traffic is modified after TLS termination.
- A **malicious client** sends crafted tensors to crash the service, consume GPU memory, bypass image safety checks, or influence model behavior. Schema and resource validation primarily address this attacker.
- A **compromised server, plugin, logger, or cache reader** can access stored embeddings or KV caches. Access controls, encryption at rest, and short retention reduce this exposure but cannot stop an authorized server process from reading data needed for inference.
- An **inversion attacker** knows the public encoder and has auxiliary image/tensor pairs, allowing it to train a decoder or classifier. This is stronger than merely seeing one unknown tensor, but realistic for open-weight encoders.

### What the architecture does provide

- The service does not routinely receive or retain the original image file.
- In the reviewed exact-string attack, Qwen3-VL, InternVL, and LLaVA encoder outputs preserve less character-level detail than encoder-free Gemma 4 and Fuyu tensors. This is model-specific evidence, not a universal guarantee.
- The client controls image preprocessing and can avoid sending metadata embedded in the source file.
- Smaller/spatially pooled representations can reduce exact OCR and pixel reconstruction leakage.

### What it does not provide

- It does not hide image semantics from the server; the server must receive enough semantics to answer.
- It does not inherently stop caption recovery, label recovery, or attribute inference, where an attacker predicts properties such as identity, location, or objects from features. It also does not inherently stop membership inference, where an attacker asks whether an example appeared in model training, or model inversion, where an attacker reconstructs input information.
- It does not provide **differential privacy**, a mathematical bound on how much one input changes an output distribution; **secure computation**, which lets a server compute without seeing plaintext values; or any information-theoretic bound on leakage.
- Public encoder weights let an attacker collect arbitrary paired images and embeddings for training an inversion model.
- Encoder-free projection models can preserve almost direct spatial samples.
- The server can potentially optimize adversarial prompts or decoders against the exact representation it receives.

### Minimum security requirements

- Use authenticated TLS 1.3 in transit. If TLS terminates before the final inference worker, bind the tensor, prompt, user, request ID, and model revision with an application-level message authentication code or signature across the internal hops.
- A versioned schema containing model ID, exact weights revision, processor revision, split point, tensor shape, dtype, quantizer/codebook version, and token ordering.
- The binary schema should also state byte order, contiguous memory layout, image-grid/position metadata, quantization scales, compression parameters, uncompressed byte limit, and sequence limit.
- Validate shape and type before GPU allocation. A finite-value check rejects `NaN` and positive/negative infinity. A norm bound rejects implausibly large vectors. These checks protect service stability but do not authenticate that an otherwise plausible tensor came from the approved encoder or a real image.
- Bind a unique request ID and expiration time to authenticated messages so an old valid tensor cannot be replayed as a new request.
- No embedding or KV logging by default; explicit short retention limits.
- Treat embeddings as sensitive personal data in access control and incident response.
- Rate limits and payload bounds to prevent GPU memory abuse.
- Safety evaluation on embedding inputs because pixel-side safety classifiers may be bypassed.
- A documented claim such as "raw-image minimization," not "the server cannot learn the image."

## 5. What are the pros and cons?

### Pros

- Raw-image data minimization and fewer server-side image copies.
- Uses client neural processing units (NPUs) or graphics processing units (GPUs) that would otherwise be idle.
- Independent scaling of vision encoding and LLM decoding.
- Query-independent encoder output can be cached across multiple questions about the same frame. Query-conditioned compressed outputs, such as QueCC, normally cannot be reused for a different question.
- Pipeline overlap can improve server throughput.
- Token pruning, quantization, and progressive transmission can eventually beat image upload.
- The client can apply local redaction, cropping, frame selection, or policy before encoding.

### Cons

- Floating-point embeddings are often larger than JPEG/WebP.
- Client compute, memory, battery, thermal load, and startup/model-download size increase.
- Model and preprocessing versions must match exactly across client and server.
- The server cannot independently re-encode or audit image preprocessing.
- Encoder updates become a distributed deployment/versioning problem.
- Embeddings remain privacy-sensitive and attackable.
- Arbitrary embeddings introduce integrity, abuse, and safety-bypass surfaces.
- Aggressive token reduction hurts OCR, small objects, spatial detail, and document tasks first.
- A wide range of client hardware makes latency and numerical equivalence harder to guarantee. Bit-identical FP16 tensors are often unrealistic across runtimes; the practical target should be equivalent generated answers or benchmark quality within a declared tolerance.
- Browser execution adds weight download, WebGPU compatibility, memory, and model-IP concerns.

## Conclusions for the project

1. **The high-level split has clear prior implementations.** This is not a patentability conclusion; it means Distributed VLMs already demonstrates complete vision encoding on edge devices and language generation on a server. A useful contribution would be a readable open-source implementation, a real privacy evaluation, and a codec whose encrypted wire bytes beat a named JPEG/WebP setting within a declared quality and client-compute budget.
2. **Start with a conventional small encoder, not Gemma 4 Unified, if exact text/pixel leakage is the main concern.** SmolVLM-256M has the strongest direct embedded demonstration reviewed here: an approximately 86M-parameter encoder, `64 x 576` output, and NXP i.MX95 measurements. Holo3.1-0.8B is the smallest current conventional encoder in the focused product-model table at about 99M parameters and is relevant for a UI/computer-use experiment. That does not prove either model will meet browser download, memory, startup, battery, or quality requirements; those still require measurement. FastVLM is another candidate if changing the complete VLM architecture and checkpoints is acceptable.
3. **Benchmark at least four wire formats:** compressed image, FP16 latent, INT8 latent, and token-reduced/quantized latent. Include TLS payload bytes, client encode time/energy, upload time, server TTFT, and task quality.
4. **Run explicit leakage tests:** caption recovery, attribute classification, OCR reconstruction, nearest-neighbor retrieval, and image inversion. Compare raw encoder output, projected output, spatial pooling, token reduction, and quantization.
5. **Use a binary protocol with authenticated encryption.** Do not ship JSON tensors or accept unvalidated arbitrary arrays.
6. **Make the split point configurable in experiments.** Pre-projector features can be narrower; post-projector features are simpler for the server. AlignedVQ shows that a very early intra-encoder split can preserve benchmark accuracy after task-aware training, but its corrected total payload is unknown. TOFC is the more defensible compression baseline because it runs the complete encoder on the device and reports entropy-coded sizes including the encoded symbols.
7. **Treat OCR/document understanding as a separate use case.** The literature consistently shows that extreme visual-token reduction works better for visual reasoning than dense text/detail tasks.

## Direct answers to the brief

| Question | Answer |
|---|---|
| What have people tried? | Whole-encoder edge splits, complete-encoder feature coding in TOFC, intra-encoder vector quantization, server-side encoder pools, token pruning/merging, progressive latent codecs, continuous feature streaming, adaptive region retransmission, routing, and speculative cooperation. |
| Latest direct attempt? | Co-VStream, first posted June 22, 2026, is the latest qualifying deployment proposal found, although captions and graph memory make it broader than a pure split. Progressive Semantic Communication, posted April 29, is the latest cleaner physical client-encoder/server-LLM testbed found. |
| Is there a platform? | No checked provider offers this as a turnkey hosted VLM API. Modal and Baseten can host a custom implementation. At the serving-engine layer, vLLM and TensorRT-LLM accept precomputed visual embeddings, while vLLM/llm-d, TensorRT-LLM, and Dynamo also provide cluster disaggregation. vLLM marks arbitrary embeddings trusted-only. |
| Open-source implementations? | vLLM and TensorRT-LLM implement direct embedding ingestion; llm-d and Dynamo implement cluster orchestration. TOFC publishes partial compression code but omits its cross-device transport. VisionZip, QueCC, and FastVLM provide related components. No complete hardened reference service intended for arbitrary untrusted Internet clients was found. |
| Pros? | Raw-image minimization, edge compute use, caching, independent scaling, pipeline overlap, and potential bandwidth savings after aggressive compression. |
| Cons? | Embeddings often exceed compressed image size, leak semantics, add a transport attack surface, require exact version coupling, and shift compute/energy to heterogeneous clients. |

## Key references

- [Distributed VLMs: Efficient Vision-Language Processing through Cloud-Edge Collaboration](https://par.nsf.gov/biblio/10639785-distributed-vlms-efficient-vision-language-processing-through-cloud-edge-collaboration)
- [Task-Oriented Feature Compression for Multimodal Understanding via Device-Edge Co-Inference](https://arxiv.org/abs/2503.12926)
- [Progressive Semantic Communication for Efficient Edge-Cloud Vision-Language Models](https://arxiv.org/abs/2604.26508)
- [Co-VStream: Edge-Cloud Collaboration for Understanding of Long Video Streams](https://arxiv.org/abs/2606.22804)
- [Aligned Vector Quantization for Edge-Cloud Collaborative Vision-Language Models](https://arxiv.org/abs/2411.05961)
- [ModServe](https://arxiv.org/abs/2502.00937)
- [VisionZip](https://arxiv.org/abs/2412.04467)
- [Inference Optimal VLMs Need Fewer Visual Tokens and More Parameters](https://arxiv.org/abs/2411.03312)
- [FastVLM](https://arxiv.org/abs/2412.13303)
- [CapRecover](https://arxiv.org/abs/2507.22828)
- [The Vision Encoder as a Privacy Boundary](https://arxiv.org/abs/2606.14783)
- [Vision Token Manipulation Attacks](https://arxiv.org/abs/2607.02819)
- [CoVSpec](https://arxiv.org/abs/2605.02218)
- [edgeVLM](https://arxiv.org/abs/2508.12638)
- [INAR-VL](https://arxiv.org/abs/2605.18853)
- [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770)
- [vLLM Disaggregated Encoder](https://docs.vllm.ai/en/latest/features/disagg_encoder/)
- [vLLM Multimodal Inputs](https://docs.vllm.ai/en/latest/features/multimodal_inputs/)
- [llm-d Encode Disaggregation](https://github.com/llm-d/llm-d/tree/main/guides/multimodal-serving/e-disaggregation)
- [TensorRT-LLM standalone multimodal encoder](https://github.com/NVIDIA/TensorRT-LLM/pull/6743)
- [TensorRT-LLM external multimodal embeddings](https://github.com/NVIDIA/TensorRT-LLM/pull/6263)
- [NVIDIA Dynamo encoder disaggregation](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/features/multimodal/encoder-disaggregation.md)
- [Modal vLLM deployment](https://modal.com/blog/how-to-deploy-vllm)
- [Baseten custom model server](https://docs.baseten.co/development/model/custom-server)
- [Baseten vision Model API](https://docs.baseten.co/inference/model-apis/vision)
- [Fireworks vision models](https://fireworksai.mintlify.app/guides/querying-vision-language-models)
- [Together AI embeddings](https://docs.together.ai/docs/inference/embeddings/embeddings)
- [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/en/index)

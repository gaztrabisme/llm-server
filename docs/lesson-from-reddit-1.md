
r/LocalLLaMA icon
Go to LocalLLaMA
r/LocalLLaMA
•
5d ago
mazuj2
[Solution Found] Qwen3-Next 80B MoE running at 39 t/s on RTX 5070 Ti + 5060 Ti (32GB VRAM)
Discussion

[Solution Found] Qwen3-Next 80B MoE running at 39 t/s on RTX 5070 Ti + 5060 Ti (32GB VRAM) - The fix nobody else figured out

Hey fellow 50 series brothers in pain,

I've been banging my head against this for a while and finally cracked it through pure trial and error. Posting this so nobody else has to suffer.

My Hardware:

RTX 5070 Ti (16GB VRAM)

RTX 5060 Ti (16GB VRAM)

32GB total VRAM

64GB System RAM

Windows 11

llama.cpp b8077 (CUDA 12.4 build)

Model: Qwen3-Next-80B-A3B-Instruct-UD-IQ2_XXS.gguf (26.2GB)

The Problem:

Out of the box, Qwen3-Next was running at 6.5 tokens/sec with:

CPU usage 25-55% going absolutely insane during thinking AND generation

GPUs sitting at 0% during thinking phase

5070 Ti at 5-10% during generation

5060 Ti at 10-40% during generation

~34GB of system RAM being consumed

Model clearly bottlenecked on CPU

Every suggestion I found online said the same generic things:

"Check your n_gpu_layers" ✅ already 999, all 49 layers on GPU

"Check your tensor split" ✅ tried everything

"Use CUDA 12.8+" ✅ not the issue

"Your offloading is broken" ❌ WRONG - layers were fully on GPU

The load output PROVED layers were on GPU:

load_tensors: offloaded 49/49 layers to GPU

load_tensors: CPU_Mapped model buffer size = 166.92 MiB (just metadata)

load_tensors: CUDA0 model buffer size = 12617.97 MiB

load_tensors: CUDA1 model buffer size = 12206.31 MiB

So why was CPU going nuts? Nobody had the right answer.

The Fix - Two flags that nobody mentioned together:

Step 1: Force ALL MoE experts off CPU

--n-cpu-moe 0

Start here. Systematically reduce from default down to 0. Each step helps. At 0 you still get CPU activity but it's better.

Step 2: THIS IS THE KEY ONE

Change from -sm row to:

-sm layer

Row-split (-sm row) splits each expert's weight matrix across both GPUs. This means every single expert call requires GPU-to-GPU communication over PCIe. For a model with 128 experts firing 8 per token, that's constant cross-GPU chatter killing your throughput.

Layer-split (-sm layer) assigns complete layers/experts to one GPU. Each GPU owns its experts fully. No cross-GPU communication during routing. The GPUs work independently and efficiently.

BOOM. 39 tokens/sec.

The Winning Command:

llama-server.exe -m Qwen3-Next-80B-A3B-Instruct-UD-IQ2_XXS.gguf -ngl 999 -c 4096 --port 8081 --n-cpu-moe 0 -t 6 -fa auto -sm layer

Results:

Before: 6.5 t/s, CPU melting, GPUs doing nothing

After: 38-39 t/s, CPUs chill, GPUs working properly

That's a 6x improvement with zero hardware changes

Why this works (the actual explanation):

Qwen3-Next uses a hybrid architecture — DeltaNet linear attention combined with high-sparsity MoE (128 experts, 8 active per token). When you row-split a MoE model across two GPUs, the expert weights are sliced horizontally across both cards. Every expert activation requires both GPUs to coordinate and combine results. With 8 experts firing per token across 47 layers, you're generating thousands of cross-GPU sync operations per token.

Layer-split instead assigns whole layers to each GPU. Experts live entirely on one card. The routing decision sends the computation to whichever GPU owns that expert. Clean, fast, no sync overhead.

Notes:

The 166MB CPU_Mapped is normal — that's just mmap metadata and tokenizer, not model weights

-t 6 sets CPU threads for the tiny bit of remaining CPU work

-fa auto enables flash attention where supported

This is on llama.cpp b8077 — make sure you're on a recent build that has Qwen3-Next support (merged in b7186)

Model fits in 32GB with ~7GB headroom for KV cache

Hope this saves someone's sanity. Took me way too long to find this and I couldn't find it documented anywhere.

If this helped you, drop a comment — curious how it performs on other 50 series configurations.

— RJ
r/LocalLLaMA - [Solution Found] Qwen3-Next 80B MoE running at 39 t/s on RTX 5070 Ti + 5060 Ti (32GB VRAM)
73
Sort by:
Comments Section
ilintar
•
5d ago
emoji:Discord:
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

Why oh why would you *ever* use `-sm row`? It's outdated and bad performance, soon to be deprecated.
24
u/Far-Low-4705 avatar
Far-Low-4705
•
5d ago
• Edited 5d ago

Doesn’t it allow for tensor parallelism?

Whenever I use it I just get garbled output so I wouldn’t know

EDIT: idk y im being downvoted, that is what it says on the official documentation...
2
ilintar
•
5d ago
emoji:Discord:
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

No. It was supposed to, but it was an abandoned attempt. True tensor parallelism is coming now.
6
u/Far-Low-4705 avatar
Far-Low-4705
•
5d ago

hm, thats interesting, good to hear!

Hopefully the overhead wont be too much for parallelism with only 2 gpus
2
p_235615
•
5d ago

Is it really worth running a 80B model at q2 quant, vs a 30B model with q4 or maybe even q6 in 32GB VRAM ? Because from what I tested, minimax-m2.5 at q2 on a RTX 6000 PRO 96GB and it wasnt very good compared to 80-120B models at q4+.
11
JsThiago5
•
5d ago

which 80-120b models did you try and recommend?
1
p_235615
•
5d ago

qwen3-code-next:80b-q6_0 for agent stuff and coding, for general questions with web search and fetch gpt-oss:120b is still quite good.

On my homeassistant AI I run gpt-oss:20b on a 16GB VRAM GPU.
1
u/m94301 avatar
m94301
•
5d ago

Great debug description and thanks for the writeup. I smiled at "Why this works". That one is copilot/GPT maybe? I see it constantly these days.
7
u/srigi avatar
srigi
•
5d ago

Let me tell you a secret… he or she didn’t wrote that piece above. The moment I saw “Why this works” I knew I’ve seen this hundreds of times on my screen.
3
_hypochonder_
•
5d ago

>Change from -sm row to:
>-sm layer
The default is layer. Row is mostly useful in larger dense models. (70+B models)
6
u/Autobahn97 avatar
Autobahn97
•
5d ago

Thanks for posting this. I just explored adding a second GPU to my PC to go from 16GB VRAM to 32GB as you have so its good to know what its capable of running if set up correctly. I had a conversation about this with Gemini and it did tell me to be certain to split layers over GPUs.
2
u/legit_split_ avatar
legit_split_
•
5d ago

Why not use --fit and --fit-ctx instead? 
4
u/mazuj2 avatar
mazuj2
OP •
5d ago

it's not a matter of the model fitting. i had 3.5gb left after forcing the whole model onto gpus but was still getting 6tok/s.
the key is Layer-split (-sm layer) assigns complete layers/experts to one GPU. Each GPU owns its experts fully. No cross-GPU communication during routing. The GPUs work independently and efficiently.
this is what i couldn't find anywhere.
5
u/legit_split_ avatar
legit_split_
•
5d ago

Sorry if I wasn't specific enough, I meant instead of trial and error with --n-cpu-moe, just use the fit parameters
1
u/HumanDrone8721 avatar
HumanDrone8721
•
5d ago
emoji:Discord:
Profile Badge for the Achievement Top 1% Commenter Top 1% Commenter

^ This. Now that llama.cpp has them and they work nicely you leave performance on the table if you don't use them and try to manually optimize.
1
overand
•
5d ago

Isn't --fit on by default? (Much like --sm layer is the default if you don't specify it?)
3
u/mazuj2 avatar
mazuj2
OP •
5d ago

tried --fit and exact same tok/s with or without.
1
u/PhilippeEiffel avatar
PhilippeEiffel
•
5d ago

Because --fit is activated by default (this is now the default value from some weeks).
2
u/ixdx avatar
ixdx
•
5d ago

I have almost the same configuration.

I'm running llama-server in docker on ubuntu 24.04 with the following parameters:

--flash-attn on --jinja --cache-ram -1 --threads -1 --cache-type-k q8_0 --cache-type-v q8_0 --device CUDA0,CUDA1 --ctx-size 131072 --model /models/unsloth/Qwen3-Coder-Next-MXFP4_MOE.gguf --temp 1.0 --top-k 40 --top-p 0.95 --fit on --fit-target 128

Here are the results of some tests:

root@04f2418c8160:/app# ./llama-bench -m /models/unsloth/Qwen3-Coder-Next-UD-Q2_K_XL.gguf 
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5070 Ti, compute capability 12.0, VMM: yes
  Device 1: NVIDIA GeForce RTX 5060 Ti, compute capability 12.0, VMM: yes
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen3next 80B.A3B Q2_K - Medium |  27.48 GiB |    79.67 B | CUDA       |  99 |           pp512 |       1013.75 ± 6.27 |
| qwen3next 80B.A3B Q2_K - Medium |  27.48 GiB |    79.67 B | CUDA       |  99 |           tg128 |         88.10 ± 0.19 |

build: 01d8eaa (1)

For Qwen3-Coder-Next-MXFP4_MOE.gguf, the test results are taken from the llama-server logs. I was unable to get llama-bench to evenly load both GPU under partial offloading to RAM.

context size: 9268
slot update_slots: id  3 | task 0 | new prompt, n_ctx_slot = 131072, n_keep = 0, task.n_tokens = 9268
prompt eval time =   44528.60 ms /  9268 tokens (    4.80 ms per token,   208.14 tokens per second)
       eval time =    1410.57 ms /    61 tokens (   23.12 ms per token,    43.24 tokens per second)
      total time =   45939.17 ms /  9329 tokens

context size: 19302
slot update_slots: id  2 | task 6909 | new prompt, n_ctx_slot = 131072, n_keep = 0, task.n_tokens = 19302
prompt eval time =    1669.86 ms /   285 tokens (    5.86 ms per token,   170.67 tokens per second)
       eval time =    8372.24 ms /   336 tokens (   24.92 ms per token,    40.13 tokens per second)
      total time =   10042.10 ms /   621 tokens

3
u/Danmoreng avatar
Danmoreng
•
5d ago

Performance under Windows tanks significantly. I get ~40t/s with Linux on a Notebook with 64GB RAM and 5080 Mobile 16GB VRAM using the mxfp4 variant (40GB model, split across gpu and cpu). With Windows it’s ~30t/s.
2
Mount_Gamer
•
5d ago

Pretty sure I get the 80b MXFP4 model running fine with the rtx5060ti 16gb and the rest on 64GB system ram, slower ecc with a 5650g pro. ~27t/s, but I'm quite new to llama.cpp, no doubt it could be better.
2
Wise_Reward6165
•
5d ago

Try using llama.cpp on Fedora or Debian and if it loads into the gpu (no cpu) you will get 150 t/s
1
u/JustSayin_thatuknow avatar
JustSayin_thatuknow
•
5d ago

Where does that number (150tps) come from?
2
u/mazuj2 avatar
mazuj2
OP •
5d ago

3 more days and bifuracation is here! 48gb and i will be running good quants of qwen3 next 80b at speed!
1
u/_-_David avatar
_-_David
•
5d ago

I am supposedly getting my new mobo delivered today so I can use my 5090 and 5060ti for 48gb of VRAM and run qwen3 coder next at q4. This will be my first time using both gpus at once, and I appreciate your post. Have fun in 3 days!
1
u/alex_godspeed avatar
alex_godspeed
•
5d ago

i got 5060ti and 9060xt (pooled 32g vram) on vulkan, and 32g ddr4

think i can do the same as yours? you have 64g sys ram =(
1
TurbulentInternet728
•
5d ago

can i run it on 4*3080?
1
u/mazuj2 avatar
mazuj2
OP •
4d ago

4 nvidia's so yes. use llama from the command line
this is what i am running for each of these models. hard to believe they would run but they do. heavy loading onto cpu ram but get good tokens/sec.
UD-IQ2_XXS 49 tokens/s

llama-server.exe -m Qwen3-Next-80B-A3B-Instruct-UD-IQ2_XXS.gguf -ngl 999 -c 4096 --port 8081 --n-cpu-moe 0 -t 12 -fa on -sm layer

Q3_K_M 22.5 tokens/s

llama-server.exe -m Qwen3-Next-80B-A3B-Instruct-Q3_K_M.gguf -ngl 41 -c 32768 --port 8081 --n-cpu-moe 12 -t 12 -fa on --tensor-split 52,48

Q4_K_M 14.69 tokens/s

llama-server.exe -m Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf -ngl 30 -c 1024 --port 8081 --n-cpu-moe 12 -t 12 -fa on --tensor-split 50,50
1
u/SimilarWarthog8393 avatar
SimilarWarthog8393
•
4d ago

This sounds wrong -- only 40 t/s when fully on GPU? With a Q5 quant of the same model I get 23 t/s on my laptop with 8gb VRAM and the rest on CPU.
1
u/Opposite-Station-337 avatar
Opposite-Station-337
•
4d ago

They are limited by the 5060ti Max token speed. I get the same tok/s with a single 5060ti 16gb and 64gb system memory. They could leave the 5060 entirely out of the equation and get more speed.
1
